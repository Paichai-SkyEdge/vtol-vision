#include "vtol_vision/yolo_detector.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <utility>

namespace vtol_vision
{

// ---------------------------------------------------------------------------
// TrtLogger
// ---------------------------------------------------------------------------

void YoloDetector::TrtLogger::log(
  nvinfer1::ILogger::Severity severity,
  const char * msg) noexcept
{
  // Suppress INFO/VERBOSE spam from TensorRT internals.
  if (severity > nvinfer1::ILogger::Severity::kWARNING) {
    return;
  }
  // Print to stderr; the ROS logger is not accessible here.
  const char * prefix =
    (severity == nvinfer1::ILogger::Severity::kERROR) ? "[TRT][ERROR]" : "[TRT][WARN]";
  std::fprintf(stderr, "%s %s\n", prefix, msg);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace
{

float ClipFloat(float v, float lo, float hi)
{
  return std::max(lo, std::min(v, hi));
}

// Read a binary file into a byte vector.
std::vector<char> ReadBinaryFile(const std::string & path)
{
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    return {};
  }
  const std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<char> buf(static_cast<std::size_t>(size));
  if (!file.read(buf.data(), size)) {
    return {};
  }
  return buf;
}

}  // namespace

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

YoloDetector::YoloDetector(rclcpp::Logger logger)
: logger_(std::move(logger))
{
}

YoloDetector::~YoloDetector()
{
  if (gpu_input_) {cudaFree(gpu_input_);}
  if (gpu_output_) {cudaFree(gpu_output_);}
  if (stream_) {cudaStreamDestroy(stream_);}

#if NV_TENSORRT_MAJOR >= 10
  delete context_;
  delete engine_;
  delete runtime_;
#else
  if (context_) {context_->destroy();}
  if (engine_) {engine_->destroy();}
  if (runtime_) {runtime_->destroy();}
#endif
}

// ---------------------------------------------------------------------------
// LoadClassMap  (unchanged from original)
// ---------------------------------------------------------------------------

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
    error_message = "failed to open class map yaml: " + yaml_path;
    return false;
  }

  cv::FileNode classes_node = fs["classes"];
  if (classes_node.type() == cv::FileNode::SEQ) {
    for (auto it = classes_node.begin(); it != classes_node.end(); ++it) {
      const cv::FileNode cls = *it;
      int id = -1;
      std::string name;
      cls["id"] >> id;
      cls["name"] >> name;
      if (id >= 0 && !name.empty()) {
        class_map[id] = name;
      }
    }
  }

  if (class_map.empty()) {
    cv::FileNode class_names_node = fs["class_names"];
    if (class_names_node.type() == cv::FileNode::SEQ) {
      int id = 0;
      for (auto it = class_names_node.begin(); it != class_names_node.end(); ++it) {
        std::string name = static_cast<std::string>(*it);
        if (!name.empty()) {
          class_map[id++] = name;
        }
      }
    }
  }

  if (class_map.empty()) {
    error_message = "no valid classes/class_names entries found in: " + yaml_path;
    return false;
  }

  error_message.clear();
  return true;
}

// ---------------------------------------------------------------------------
// Initialize
// ---------------------------------------------------------------------------

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
  conf_threshold_ = std::max(0.0F, conf_threshold);
  nms_threshold_ = std::max(0.0F, nms_threshold);

  if (model_path.empty()) {
    RCLCPP_WARN(logger_, "trt_engine_path is empty. YOLO inference disabled.");
    return false;
  }
  if (!std::filesystem::exists(model_path)) {
    RCLCPP_ERROR(logger_, "engine file does not exist: %s", model_path.c_str());
    return false;
  }
  if (std::filesystem::path(model_path).extension() != ".engine") {
    RCLCPP_ERROR(
      logger_,
      "expected a .engine file, got: %s",
      model_path.c_str());
    return false;
  }

  // --- Deserialize engine ---
  const std::vector<char> engine_data = ReadBinaryFile(model_path);
  if (engine_data.empty()) {
    RCLCPP_ERROR(logger_, "failed to read engine file: %s", model_path.c_str());
    return false;
  }

  runtime_ = nvinfer1::createInferRuntime(trt_logger_);
  if (!runtime_) {
    RCLCPP_ERROR(logger_, "failed to create TensorRT runtime");
    return false;
  }

  engine_ = runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size());
  if (!engine_) {
    RCLCPP_ERROR(logger_, "failed to deserialize engine: %s", model_path.c_str());
    return false;
  }

  context_ = engine_->createExecutionContext();
  if (!context_) {
    RCLCPP_ERROR(logger_, "failed to create execution context");
    return false;
  }

  // --- Discover I/O tensor shapes ---
  const std::size_t input_bytes =
    static_cast<std::size_t>(input_size_) *
    static_cast<std::size_t>(input_size_) * 3U * sizeof(float);

  std::size_t output_bytes = 0;

#if NV_TENSORRT_MAJOR >= 10
  // TensorRT 10+ API
  const int n_io = engine_->getNbIOTensors();
  for (int i = 0; i < n_io; ++i) {
    const char * name = engine_->getIOTensorName(i);
    const nvinfer1::TensorIOMode mode = engine_->getTensorIOMode(name);
    const nvinfer1::Dims dims = engine_->getTensorShape(name);

    if (mode == nvinfer1::TensorIOMode::kINPUT) {
      // expected [1, 3, H, W]
      (void)dims;
    } else {
      // expected [1, num_fields, num_candidates]
      if (dims.nbDims == 3) {
        num_fields_ = static_cast<int>(dims.d[1]);
        num_candidates_ = static_cast<int>(dims.d[2]);
      }
      output_bytes =
        static_cast<std::size_t>(num_fields_) *
        static_cast<std::size_t>(num_candidates_) * sizeof(float);
    }
  }
#else
  // TensorRT 8 API
  const int n_bindings = engine_->getNbBindings();
  for (int i = 0; i < n_bindings; ++i) {
    const nvinfer1::Dims dims = engine_->getBindingDimensions(i);
    if (engine_->bindingIsInput(i)) {
      input_binding_idx_ = i;
    } else {
      output_binding_idx_ = i;
      if (dims.nbDims == 3) {
        num_fields_ = static_cast<int>(dims.d[1]);
        num_candidates_ = static_cast<int>(dims.d[2]);
      }
      output_bytes =
        static_cast<std::size_t>(num_fields_) *
        static_cast<std::size_t>(num_candidates_) * sizeof(float);
    }
  }
#endif

  if (num_candidates_ <= 0 || num_fields_ < 5) {
    RCLCPP_ERROR(
      logger_,
      "unexpected engine output shape: num_fields=%d num_candidates=%d",
      num_fields_,
      num_candidates_);
    return false;
  }

  // --- Allocate GPU buffers ---
  if (cudaMalloc(&gpu_input_, input_bytes) != cudaSuccess ||
    cudaMalloc(&gpu_output_, output_bytes) != cudaSuccess)
  {
    RCLCPP_ERROR(logger_, "cudaMalloc failed for I/O buffers");
    return false;
  }

  if (cudaStreamCreate(&stream_) != cudaSuccess) {
    RCLCPP_ERROR(logger_, "cudaStreamCreate failed");
    return false;
  }

  cpu_input_.resize(3 * input_size_ * input_size_);
  cpu_output_.resize(static_cast<std::size_t>(num_fields_) *
    static_cast<std::size_t>(num_candidates_));

#if NV_TENSORRT_MAJOR >= 10
  // Bind tensor addresses once — context retains them across inferences.
  const int n_io2 = engine_->getNbIOTensors();
  for (int i = 0; i < n_io2; ++i) {
    const char * name = engine_->getIOTensorName(i);
    const nvinfer1::TensorIOMode mode = engine_->getTensorIOMode(name);
    void * ptr = (mode == nvinfer1::TensorIOMode::kINPUT) ? gpu_input_ : gpu_output_;
    context_->setTensorAddress(name, ptr);
  }
#endif

  is_ready_ = true;
  RCLCPP_INFO(
    logger_,
    "TensorRT engine loaded: %s  output=[1,%d,%d]  conf=%.2f  nms=%.2f",
    model_path.c_str(),
    num_fields_,
    num_candidates_,
    static_cast<double>(conf_threshold_),
    static_cast<double>(nms_threshold_));
  return true;
}

// ---------------------------------------------------------------------------
// Infer
// ---------------------------------------------------------------------------

std::vector<ObjectCandidate> YoloDetector::Infer(const cv::Mat & frame)
{
  if (!is_ready_ || frame.empty()) {
    return {};
  }

  LetterboxMeta meta;
  const cv::Mat letterboxed = PrepareInput(frame, meta);
  if (letterboxed.empty()) {
    return {};
  }

  // BGR → RGB, uint8 → float32 [0,1], HWC → CHW
  cv::Mat rgb;
  cv::cvtColor(letterboxed, rgb, cv::COLOR_BGR2RGB);
  cv::Mat float_img;
  rgb.convertTo(float_img, CV_32F, 1.0 / 255.0);

  std::vector<cv::Mat> channels(3);
  cv::split(float_img, channels);
  const int plane = input_size_ * input_size_;
  for (int c = 0; c < 3; ++c) {
    std::memcpy(cpu_input_.data() + c * plane, channels[c].data, plane * sizeof(float));
  }

  // Host → Device
  const std::size_t input_bytes = cpu_input_.size() * sizeof(float);
  if (cudaMemcpyAsync(
      gpu_input_,
      cpu_input_.data(),
      input_bytes,
      cudaMemcpyHostToDevice,
      stream_) != cudaSuccess)
  {
    RCLCPP_WARN(logger_, "cudaMemcpyAsync H2D failed");
    return {};
  }

  // Inference
  bool ok = false;
#if NV_TENSORRT_MAJOR >= 10
  ok = context_->enqueueV3(stream_);
#else
  void * bindings[2];
  bindings[input_binding_idx_] = gpu_input_;
  bindings[output_binding_idx_] = gpu_output_;
  ok = context_->enqueueV2(bindings, stream_, nullptr);
#endif
  if (!ok) {
    RCLCPP_WARN(logger_, "TensorRT enqueue failed");
    return {};
  }

  // Device → Host
  const std::size_t output_bytes = cpu_output_.size() * sizeof(float);
  if (cudaMemcpyAsync(
      cpu_output_.data(),
      gpu_output_,
      output_bytes,
      cudaMemcpyDeviceToHost,
      stream_) != cudaSuccess)
  {
    RCLCPP_WARN(logger_, "cudaMemcpyAsync D2H failed");
    return {};
  }

  cudaStreamSynchronize(stream_);

  return PostProcess(
    cpu_output_.data(),
    num_candidates_,
    num_fields_,
    meta,
    frame.size());
}

// ---------------------------------------------------------------------------
// PrepareInput  (letterbox)
// ---------------------------------------------------------------------------

cv::Mat YoloDetector::PrepareInput(const cv::Mat & frame, LetterboxMeta & meta) const
{
  cv::Mat bgr;
  if (frame.channels() == 1) {
    cv::cvtColor(frame, bgr, cv::COLOR_GRAY2BGR);
  } else {
    bgr = frame;
  }

  const float scale = std::min(
    static_cast<float>(input_size_) / static_cast<float>(bgr.cols),
    static_cast<float>(input_size_) / static_cast<float>(bgr.rows));

  const int rw = static_cast<int>(std::round(bgr.cols * scale));
  const int rh = static_cast<int>(std::round(bgr.rows * scale));
  const int pad_x = (input_size_ - rw) / 2;
  const int pad_y = (input_size_ - rh) / 2;

  cv::Mat resized;
  cv::resize(bgr, resized, cv::Size(rw, rh));

  cv::Mat out(
    input_size_, input_size_, CV_8UC3,
    cv::Scalar(114, 114, 114));
  resized.copyTo(out(cv::Rect(pad_x, pad_y, rw, rh)));

  meta.scale = scale;
  meta.pad_x = pad_x;
  meta.pad_y = pad_y;
  return out;
}

// ---------------------------------------------------------------------------
// PostProcess
//
// Output tensor layout: [1, num_fields, num_candidates]  (row-major)
//   → data[field * num_candidates + candidate_idx]
//
// Ultralytics YOLO single-class:  num_fields = 5  (cx, cy, w, h, score)
// Multi-class:                    num_fields = 4 + N
// ---------------------------------------------------------------------------

std::vector<ObjectCandidate> YoloDetector::PostProcess(
  const float * output,
  int num_candidates,
  int num_fields,
  const LetterboxMeta & meta,
  const cv::Size & original_size) const
{
  const int num_classes = num_fields - 4;

  std::vector<cv::Rect> boxes;
  std::vector<float> scores;
  std::vector<int> class_ids;

  for (int i = 0; i < num_candidates; ++i) {
    // Access each field for this candidate via column-major stride.
    const float cx = output[0 * num_candidates + i];
    const float cy = output[1 * num_candidates + i];
    const float w = output[2 * num_candidates + i];
    const float h = output[3 * num_candidates + i];

    int best_class = 0;
    float confidence = 0.0F;

    if (num_classes == 1) {
      confidence = output[4 * num_candidates + i];
      best_class = 0;
    } else {
      for (int c = 0; c < num_classes; ++c) {
        const float s = output[(4 + c) * num_candidates + i];
        if (s > confidence) {
          confidence = s;
          best_class = c;
        }
      }
    }

    if (confidence < conf_threshold_) {
      continue;
    }

    // Undo letterbox: map from padded input space back to original image.
    const float x0 = (cx - w * 0.5F - static_cast<float>(meta.pad_x)) / meta.scale;
    const float y0 = (cy - h * 0.5F - static_cast<float>(meta.pad_y)) / meta.scale;
    const float bw = w / meta.scale;
    const float bh = h / meta.scale;

    cv::Rect box(
      static_cast<int>(std::round(x0)),
      static_cast<int>(std::round(y0)),
      static_cast<int>(std::round(bw)),
      static_cast<int>(std::round(bh)));
    box = ClipRect(box, original_size);
    if (box.area() <= 0) {
      continue;
    }

    boxes.push_back(box);
    scores.push_back(confidence);
    class_ids.push_back(best_class);
  }

  std::vector<int> kept;
  cv::dnn::NMSBoxes(boxes, scores, conf_threshold_, nms_threshold_, kept);

  std::vector<ObjectCandidate> results;
  results.reserve(kept.size());
  for (const int idx : kept) {
    ObjectCandidate c;
    c.class_id = class_ids[idx];
    c.class_name = ResolveClassName(c.class_id);
    c.score = scores[idx];
    c.bbox = boxes[idx];
    results.push_back(std::move(c));
  }
  return results;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

std::string YoloDetector::ResolveClassName(int class_id) const
{
  const auto it = class_map_.find(class_id);
  return (it != class_map_.end()) ? it->second : "class_" + std::to_string(class_id);
}

cv::Rect YoloDetector::ClipRect(const cv::Rect & box, const cv::Size & frame_size) const
{
  const int x = std::max(0, box.x);
  const int y = std::max(0, box.y);
  const int w = std::max(0, std::min(box.width, frame_size.width - x));
  const int h = std::max(0, std::min(box.height, frame_size.height - y));
  return cv::Rect(x, y, w, h);
}

}  // namespace vtol_vision
