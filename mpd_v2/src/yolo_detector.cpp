#include "vtol_vision/yolo_detector.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace vtol_vision
{

// =========================================================================
// Constructor / Destructor
// =========================================================================

YoloDetector::YoloDetector() = default;

YoloDetector::~YoloDetector()
{
#ifdef HAS_TENSORRT
  if (gpu_input_) cudaFree(gpu_input_);
  if (gpu_output_) cudaFree(gpu_output_);
  if (stream_) cudaStreamDestroy(stream_);
  if (context_) {
#if NV_TENSORRT_MAJOR >= 10
    delete context_;
#else
    context_->destroy();
#endif
  }
  if (engine_) {
#if NV_TENSORRT_MAJOR >= 10
    delete engine_;
#else
    engine_->destroy();
#endif
  }
  if (runtime_) {
#if NV_TENSORRT_MAJOR >= 10
    delete runtime_;
#else
    runtime_->destroy();
#endif
  }
#endif
}

// =========================================================================
// LoadClassMap
// =========================================================================

bool YoloDetector::LoadClassMap(
  const std::string & yaml_path,
  std::unordered_map<int, std::string> & class_map,
  std::string & error_message)
{
  class_map.clear();
  if (yaml_path.empty()) {
    error_message = "class_map_yaml is empty";
    return false;
  }
  cv::FileStorage fs(yaml_path, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    error_message = "failed to open: " + yaml_path;
    return false;
  }
  cv::FileNode classes_node = fs["classes"];
  if (classes_node.type() == cv::FileNode::SEQ) {
    for (auto it = classes_node.begin(); it != classes_node.end(); ++it) {
      int id = -1;
      std::string name;
      (*it)["id"] >> id;
      (*it)["name"] >> name;
      if (id >= 0 && !name.empty()) class_map[id] = name;
    }
  }
  if (class_map.empty()) {
    cv::FileNode cnames = fs["class_names"];
    if (cnames.type() == cv::FileNode::SEQ) {
      int id = 0;
      for (auto it = cnames.begin(); it != cnames.end(); ++it) {
        std::string n = static_cast<std::string>(*it);
        if (!n.empty()) class_map[id++] = n;
      }
    }
  }
  if (class_map.empty()) {
    error_message = "no classes found in: " + yaml_path;
    return false;
  }
  return true;
}

// =========================================================================
// Initialize — auto-detect backend
// =========================================================================

bool YoloDetector::Initialize(
  const std::string & model_path,
  const std::unordered_map<int, std::string> & class_map,
  int input_size,
  float conf_threshold,
  float nms_threshold)
{
  is_ready_ = false;
  class_map_ = class_map;
  input_size_ = std::max(32, input_size);
  conf_threshold_ = conf_threshold;
  nms_threshold_ = nms_threshold;

  if (model_path.empty()) {
    std::cerr << "[YOLO] model path empty" << std::endl;
    return false;
  }
  if (!std::filesystem::exists(model_path)) {
    std::cerr << "[YOLO] model not found: " << model_path << std::endl;
    return false;
  }

  std::string ext = std::filesystem::path(model_path).extension().string();
  if (ext == ".engine") {
#ifdef HAS_TENSORRT
    backend_ = Backend::TensorRT;
    return InitTRT(model_path);
#else
    std::cerr << "[YOLO] TensorRT not compiled in. Use ONNX instead." << std::endl;
    return false;
#endif
  } else if (ext == ".onnx") {
    backend_ = Backend::DNN;
    return InitDNN(model_path);
  } else {
    // Try DNN anyway (OpenCV can load various formats)
    backend_ = Backend::DNN;
    return InitDNN(model_path);
  }
}

// =========================================================================
// Infer — dispatch to active backend
// =========================================================================

std::vector<ObjectCandidate> YoloDetector::Infer(const cv::Mat & frame)
{
  if (!is_ready_ || frame.empty()) return {};
  switch (backend_) {
#ifdef HAS_TENSORRT
    case Backend::TensorRT: return InferTRT(frame);
#endif
    default: return InferDNN(frame);
  }
}

// =========================================================================
// DNN (ONNX) Backend
// =========================================================================

bool YoloDetector::InitDNN(const std::string & model_path)
{
  try {
    dnn_net_ = cv::dnn::readNet(model_path);
  } catch (const cv::Exception & e) {
    std::cerr << "[YOLO] DNN load failed: " << e.what() << std::endl;
    return false;
  }

  // Prefer CUDA if available, fallback to CPU
  try {
    dnn_net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    dnn_net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    std::cout << "[YOLO] DNN: CUDA backend" << std::endl;
  } catch (const cv::Exception &) {
    dnn_net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    dnn_net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    std::cout << "[YOLO] DNN: CPU backend" << std::endl;
  }

  is_ready_ = true;
  std::cout << "[YOLO] ONNX loaded: " << model_path
            << "  conf=" << conf_threshold_ << " nms=" << nms_threshold_ << std::endl;
  return true;
}

cv::Mat YoloDetector::PrepareInputDNN(const cv::Mat & frame, LetterboxMeta & meta) const
{
  cv::Mat bgr = (frame.channels() == 1) ? ([&]() {
    cv::Mat tmp; cv::cvtColor(frame, tmp, cv::COLOR_GRAY2BGR); return tmp; })() : frame;

  const float scale = std::min(
    static_cast<float>(input_size_) / bgr.cols,
    static_cast<float>(input_size_) / bgr.rows);
  const int rw = static_cast<int>(std::round(bgr.cols * scale));
  const int rh = static_cast<int>(std::round(bgr.rows * scale));
  meta.pad_x = (input_size_ - rw) / 2;
  meta.pad_y = (input_size_ - rh) / 2;
  meta.scale = scale;

  cv::Mat resized, out;
  cv::resize(bgr, resized, cv::Size(rw, rh));
  cv::copyMakeBorder(resized, out, meta.pad_y, input_size_ - rh - meta.pad_y,
                     meta.pad_x, input_size_ - rw - meta.pad_x,
                     cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
  return out;
}

std::vector<ObjectCandidate> YoloDetector::PostProcessDNN(
  const cv::Mat & output,
  const LetterboxMeta & meta,
  const cv::Size & original_size) const
{
  // Output shape: [1, 5+num_classes, 8400] or [1, 5, 8400] for single-class
  // OpenCV DNN outputs [1, num_fields, num_candidates] as Mat
  const int num_fields = output.size[1];
  const int num_candidates = output.size[2];
  const float * data = reinterpret_cast<const float *>(output.data);

  std::vector<cv::Rect> boxes;
  std::vector<float> scores;
  std::vector<int> class_ids;

  for (int i = 0; i < num_candidates; ++i) {
    float cx = data[0 * num_candidates + i];
    float cy = data[1 * num_candidates + i];
    float w  = data[2 * num_candidates + i];
    float h  = data[3 * num_candidates + i];

    int best_class = 0;
    float confidence = 0.0f;
    int nc = num_fields - 4;
    if (nc == 1) {
      confidence = data[4 * num_candidates + i];
    } else {
      for (int c = 0; c < nc; ++c) {
        float s = data[(4 + c) * num_candidates + i];
        if (s > confidence) { confidence = s; best_class = c; }
      }
    }
    if (confidence < conf_threshold_) continue;

    float x0 = (cx - w * 0.5f - meta.pad_x) / meta.scale;
    float y0 = (cy - h * 0.5f - meta.pad_y) / meta.scale;
    float bw = w / meta.scale;
    float bh = h / meta.scale;

    cv::Rect b(static_cast<int>(std::round(x0)), static_cast<int>(std::round(y0)),
               static_cast<int>(std::round(bw)), static_cast<int>(std::round(bh)));
    b = ClipRect(b, original_size);
    if (b.area() <= 0) continue;

    boxes.push_back(b);
    scores.push_back(confidence);
    class_ids.push_back(best_class);
  }

  std::vector<int> kept;
  cv::dnn::NMSBoxes(boxes, scores, conf_threshold_, nms_threshold_, kept);

  std::vector<ObjectCandidate> results;
  for (int idx : kept) {
    results.push_back({class_ids[idx], ResolveClassName(class_ids[idx]), scores[idx], boxes[idx]});
  }
  return results;
}

std::vector<ObjectCandidate> YoloDetector::InferDNN(const cv::Mat & frame)
{
  LetterboxMeta meta;
  cv::Mat blob = cv::dnn::blobFromImage(PrepareInputDNN(frame, meta), 1.0 / 255.0,
                                         cv::Size(input_size_, input_size_),
                                         cv::Scalar(), true, false);
  dnn_net_.setInput(blob);
  cv::Mat output = dnn_net_.forward();
  return PostProcessDNN(output, meta, frame.size());
}

// =========================================================================
// TensorRT Backend (Jetson-only, compiled with -DHAS_TENSORRT)
// =========================================================================

#ifdef HAS_TENSORRT

void YoloDetector::TrtLogger::log(nvinfer1::ILogger::Severity severity, const char * msg) noexcept
{
  if (severity > nvinfer1::ILogger::Severity::kWARNING) return;
  std::cerr << (severity == nvinfer1::ILogger::Severity::kERROR ? "[TRT][E] " : "[TRT][W] ")
            << msg << std::endl;
}

namespace {

std::vector<char> ReadBinaryFile(const std::string & path)
{
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) return {};
  std::streamsize sz = f.tellg();
  f.seekg(0, std::ios::beg);
  std::vector<char> buf(static_cast<std::size_t>(sz));
  f.read(buf.data(), sz);
  return f ? buf : std::vector<char>{};
}

}  // namespace

bool YoloDetector::InitTRT(const std::string & model_path)
{
  auto data = ReadBinaryFile(model_path);
  if (data.empty()) { std::cerr << "[TRT] read failed: " << model_path << std::endl; return false; }

  runtime_ = nvinfer1::createInferRuntime(trt_logger_);
  if (!runtime_) { std::cerr << "[TRT] createInferRuntime failed" << std::endl; return false; }

  engine_ = runtime_->deserializeCudaEngine(data.data(), data.size());
  if (!engine_) { std::cerr << "[TRT] deserialize failed" << std::endl; return false; }

  context_ = engine_->createExecutionContext();
  if (!context_) { std::cerr << "[TRT] createContext failed" << std::endl; return false; }

  std::size_t input_bytes = static_cast<std::size_t>(input_size_) * input_size_ * 3 * sizeof(float);
  std::size_t output_bytes = 0;

#if NV_TENSORRT_MAJOR >= 10
  for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
    const char * name = engine_->getIOTensorName(i);
    auto mode = engine_->getTensorIOMode(name);
    auto dims = engine_->getTensorShape(name);
    if (mode == nvinfer1::TensorIOMode::kINPUT) {
      (void)dims;
    } else if (dims.nbDims == 3) {
      num_fields_ = dims.d[1];
      num_candidates_ = dims.d[2];
      output_bytes = static_cast<std::size_t>(num_fields_) * num_candidates_ * sizeof(float);
    }
  }
#else
  for (int i = 0; i < engine_->getNbBindings(); ++i) {
    auto dims = engine_->getBindingDimensions(i);
    if (engine_->bindingIsInput(i)) {
      input_binding_idx_ = i;
    } else {
      output_binding_idx_ = i;
      if (dims.nbDims == 3) {
        num_fields_ = dims.d[1];
        num_candidates_ = dims.d[2];
        output_bytes = static_cast<std::size_t>(num_fields_) * num_candidates_ * sizeof(float);
      }
    }
  }
#endif

  if (num_candidates_ <= 0 || num_fields_ < 5) {
    std::cerr << "[TRT] bad output shape: " << num_fields_ << "x" << num_candidates_ << std::endl;
    return false;
  }
  if (cudaMalloc(&gpu_input_, input_bytes) != cudaSuccess ||
      cudaMalloc(&gpu_output_, output_bytes) != cudaSuccess ||
      cudaStreamCreate(&stream_) != cudaSuccess) {
    std::cerr << "[TRT] alloc failed" << std::endl;
    return false;
  }
  cpu_input_.resize(3 * input_size_ * input_size_);
  cpu_output_.resize(static_cast<std::size_t>(num_fields_) * num_candidates_);

#if NV_TENSORRT_MAJOR >= 10
  for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
    const char * name = engine_->getIOTensorName(i);
    void * ptr = (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) ? gpu_input_ : gpu_output_;
    context_->setTensorAddress(name, ptr);
  }
#endif

  is_ready_ = true;
  std::cout << "[YOLO] TRT engine: " << model_path << " [" << num_fields_ << "," << num_candidates_ << "]" << std::endl;
  return true;
}

cv::Mat YoloDetector::PrepareInputTRT(const cv::Mat & frame, LetterboxMeta & meta) const
{
  cv::Mat bgr = (frame.channels() == 1) ? ([&]() {
    cv::Mat tmp; cv::cvtColor(frame, tmp, cv::COLOR_GRAY2BGR); return tmp; })() : frame;

  float scale = std::min(
    static_cast<float>(input_size_) / bgr.cols,
    static_cast<float>(input_size_) / bgr.rows);
  int rw = static_cast<int>(std::round(bgr.cols * scale));
  int rh = static_cast<int>(std::round(bgr.rows * scale));
  meta.pad_x = (input_size_ - rw) / 2;
  meta.pad_y = (input_size_ - rh) / 2;
  meta.scale = scale;

  cv::Mat resized;
  cv::resize(bgr, resized, cv::Size(rw, rh));
  cv::Mat out(input_size_, input_size_, CV_8UC3, cv::Scalar(114, 114, 114));
  resized.copyTo(out(cv::Rect(meta.pad_x, meta.pad_y, rw, rh)));
  return out;
}

std::vector<ObjectCandidate> YoloDetector::PostProcessTRT(
  const float * output, int num_candidates, int num_fields,
  const LetterboxMeta & meta, const cv::Size & original_size) const
{
  int nc = num_fields - 4;
  std::vector<cv::Rect> boxes;
  std::vector<float> scores;
  std::vector<int> class_ids;

  for (int i = 0; i < num_candidates; ++i) {
    float cx = output[0 * num_candidates + i];
    float cy = output[1 * num_candidates + i];
    float w  = output[2 * num_candidates + i];
    float h  = output[3 * num_candidates + i];
    int best = 0;
    float conf = 0.0f;
    if (nc == 1) {
      conf = output[4 * num_candidates + i];
    } else {
      for (int c = 0; c < nc; ++c) {
        float s = output[(4 + c) * num_candidates + i];
        if (s > conf) { conf = s; best = c; }
      }
    }
    if (conf < conf_threshold_) continue;
    float x0 = (cx - w * 0.5f - meta.pad_x) / meta.scale;
    float y0 = (cy - h * 0.5f - meta.pad_y) / meta.scale;
    cv::Rect b(static_cast<int>(std::round(x0)), static_cast<int>(std::round(y0)),
               static_cast<int>(std::round(w / meta.scale)), static_cast<int>(std::round(h / meta.scale)));
    b = ClipRect(b, original_size);
    if (b.area() <= 0) continue;
    boxes.push_back(b); scores.push_back(conf); class_ids.push_back(best);
  }
  std::vector<int> kept;
  cv::dnn::NMSBoxes(boxes, scores, conf_threshold_, nms_threshold_, kept);
  std::vector<ObjectCandidate> results;
  for (int idx : kept) {
    results.push_back({class_ids[idx], ResolveClassName(class_ids[idx]), scores[idx], boxes[idx]});
  }
  return results;
}

std::vector<ObjectCandidate> YoloDetector::InferTRT(const cv::Mat & frame)
{
  LetterboxMeta meta;
  cv::Mat lb = PrepareInputTRT(frame, meta);
  if (lb.empty()) return {};

  cv::Mat rgb;
  cv::cvtColor(lb, rgb, cv::COLOR_BGR2RGB);
  cv::Mat flt;
  rgb.convertTo(flt, CV_32F, 1.0 / 255.0);
  std::vector<cv::Mat> chans(3);
  cv::split(flt, chans);
  int plane = input_size_ * input_size_;
  for (int c = 0; c < 3; ++c) {
    std::memcpy(cpu_input_.data() + c * plane, chans[c].data, plane * sizeof(float));
  }

  std::size_t ib = cpu_input_.size() * sizeof(float);
  if (cudaMemcpyAsync(gpu_input_, cpu_input_.data(), ib, cudaMemcpyHostToDevice, stream_) != cudaSuccess)
    return {};
  bool ok = false;
#if NV_TENSORRT_MAJOR >= 10
  ok = context_->enqueueV3(stream_);
#else
  void * bindings[2] = {nullptr, nullptr};
  bindings[input_binding_idx_] = gpu_input_;
  bindings[output_binding_idx_] = gpu_output_;
  ok = context_->enqueueV2(bindings, stream_, nullptr);
#endif
  if (!ok) return {};
  std::size_t ob = cpu_output_.size() * sizeof(float);
  if (cudaMemcpyAsync(cpu_output_.data(), gpu_output_, ob, cudaMemcpyDeviceToHost, stream_) != cudaSuccess)
    return {};
  cudaStreamSynchronize(stream_);
  return PostProcessTRT(cpu_output_.data(), num_candidates_, num_fields_, meta, frame.size());
}

#endif  // HAS_TENSORRT

// =========================================================================
// Shared helpers
// =========================================================================

std::string YoloDetector::ResolveClassName(int class_id) const
{
  auto it = class_map_.find(class_id);
  return (it != class_map_.end()) ? it->second : "class_" + std::to_string(class_id);
}

cv::Rect YoloDetector::ClipRect(const cv::Rect & box, const cv::Size & frame_size) const
{
  return cv::Rect(
    std::max(0, box.x), std::max(0, box.y),
    std::max(0, std::min(box.width, frame_size.width - box.x)),
    std::max(0, std::min(box.height, frame_size.height - box.y)));
}

}  // namespace vtol_vision
