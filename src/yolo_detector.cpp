#include "vtol_vision/yolo_detector.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <utility>

namespace vtol_vision
{

namespace
{
constexpr float kLetterboxPaddingValue = 114.0F;

float ClipFloat(float value, float min_value, float max_value)
{
  return std::max(min_value, std::min(value, max_value));
}
}  // namespace

YoloDetector::YoloDetector(rclcpp::Logger logger)
: logger_(std::move(logger))
{
}

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
        const cv::FileNode name_node = *it;
        std::string name = static_cast<std::string>(name_node);
        if (!name.empty()) {
          class_map[id] = name;
        }
        ++id;
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
    RCLCPP_ERROR(logger_, "model path does not exist: %s", model_path.c_str());
    return false;
  }

  try {
    net_ = cv::dnn::readNet(model_path);
  } catch (const cv::Exception & ex) {
    RCLCPP_ERROR(logger_, "failed to load model from %s: %s", model_path.c_str(), ex.what());
    return false;
  }

  if (net_.empty()) {
    RCLCPP_ERROR(logger_, "loaded network is empty: %s", model_path.c_str());
    return false;
  }

  try {
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
    RCLCPP_INFO(logger_, "YOLO backend: CUDA FP16");
  } catch (const cv::Exception &) {
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    RCLCPP_WARN(
      logger_,
      "CUDA backend unavailable. Falling back to OpenCV CPU backend.");
  }

  is_ready_ = true;
  return true;
}

std::vector<ObjectCandidate> YoloDetector::Infer(const cv::Mat & frame)
{
  if (!is_ready_ || frame.empty()) {
    return {};
  }

  LetterboxMeta meta;
  const cv::Mat input = PrepareInput(frame, meta);
  if (input.empty()) {
    return {};
  }

  cv::Mat blob = cv::dnn::blobFromImage(
    input,
    1.0 / 255.0,
    cv::Size(input_size_, input_size_),
    cv::Scalar(),
    true,
    false);

  net_.setInput(blob);
  std::vector<cv::Mat> outputs;
  net_.forward(outputs, net_.getUnconnectedOutLayersNames());
  if (outputs.empty()) {
    return {};
  }

  cv::Mat detections;
  const cv::Mat & out = outputs.front();
  if (out.dims == 3) {
    const int rows = out.size[1];
    const int cols = out.size[2];
    detections = cv::Mat(rows, cols, CV_32F, const_cast<float *>(out.ptr<float>()));
  } else if (out.dims == 2) {
    detections = out;
  } else {
    RCLCPP_WARN(logger_, "unsupported YOLO output rank: %d", out.dims);
    return {};
  }

  if (detections.cols > 6) {
    return ParseCenterBoxOutput(detections, meta, frame.size());
  }
  return ParseCornerBoxOutput(detections, meta, frame.size());
}

cv::Mat YoloDetector::PrepareInput(const cv::Mat & frame, LetterboxMeta & meta) const
{
  cv::Mat bgr_frame;
  if (frame.channels() == 1) {
    cv::cvtColor(frame, bgr_frame, cv::COLOR_GRAY2BGR);
  } else {
    bgr_frame = frame;
  }

  const float scale = std::min(
    static_cast<float>(input_size_) / static_cast<float>(bgr_frame.cols),
    static_cast<float>(input_size_) / static_cast<float>(bgr_frame.rows));

  const int resized_w = static_cast<int>(std::round(bgr_frame.cols * scale));
  const int resized_h = static_cast<int>(std::round(bgr_frame.rows * scale));
  const int pad_x = (input_size_ - resized_w) / 2;
  const int pad_y = (input_size_ - resized_h) / 2;

  cv::Mat resized;
  cv::resize(bgr_frame, resized, cv::Size(resized_w, resized_h));

  cv::Mat letterboxed(
    input_size_,
    input_size_,
    CV_8UC3,
    cv::Scalar(kLetterboxPaddingValue, kLetterboxPaddingValue, kLetterboxPaddingValue));
  resized.copyTo(letterboxed(cv::Rect(pad_x, pad_y, resized_w, resized_h)));

  meta.scale = scale;
  meta.pad_x = pad_x;
  meta.pad_y = pad_y;
  return letterboxed;
}

std::vector<ObjectCandidate> YoloDetector::ParseCenterBoxOutput(
  const cv::Mat & detections,
  const LetterboxMeta & meta,
  const cv::Size & original_size) const
{
  std::vector<cv::Rect> candidate_boxes;
  std::vector<float> candidate_scores;
  std::vector<int> candidate_class_ids;

  for (int row = 0; row < detections.rows; ++row) {
    const float * data = detections.ptr<float>(row);
    const float objectness = data[4];
    if (objectness <= 0.0F) {
      continue;
    }

    int best_class = -1;
    float best_class_score = 0.0F;
    for (int cls_idx = 5; cls_idx < detections.cols; ++cls_idx) {
      const float score = data[cls_idx];
      if (score > best_class_score) {
        best_class_score = score;
        best_class = cls_idx - 5;
      }
    }

    const float confidence = objectness * best_class_score;
    if (best_class < 0 || confidence < conf_threshold_) {
      continue;
    }

    const float cx = data[0];
    const float cy = data[1];
    const float w = data[2];
    const float h = data[3];

    const float x0 = (cx - (w * 0.5F) - static_cast<float>(meta.pad_x)) / meta.scale;
    const float y0 = (cy - (h * 0.5F) - static_cast<float>(meta.pad_y)) / meta.scale;
    const float ww = w / meta.scale;
    const float hh = h / meta.scale;

    cv::Rect box(
      static_cast<int>(std::round(x0)),
      static_cast<int>(std::round(y0)),
      static_cast<int>(std::round(ww)),
      static_cast<int>(std::round(hh)));
    box = ClipRect(box, original_size);
    if (box.area() <= 0) {
      continue;
    }

    candidate_boxes.push_back(box);
    candidate_scores.push_back(confidence);
    candidate_class_ids.push_back(best_class);
  }

  std::vector<int> kept_indices;
  cv::dnn::NMSBoxes(candidate_boxes, candidate_scores, conf_threshold_, nms_threshold_, kept_indices);

  std::vector<ObjectCandidate> results;
  results.reserve(kept_indices.size());
  for (int index : kept_indices) {
    ObjectCandidate candidate;
    candidate.class_id = candidate_class_ids[index];
    candidate.class_name = ResolveClassName(candidate.class_id);
    candidate.score = candidate_scores[index];
    candidate.bbox = candidate_boxes[index];
    results.push_back(std::move(candidate));
  }
  return results;
}

std::vector<ObjectCandidate> YoloDetector::ParseCornerBoxOutput(
  const cv::Mat & detections,
  const LetterboxMeta & meta,
  const cv::Size & original_size) const
{
  std::vector<cv::Rect> candidate_boxes;
  std::vector<float> candidate_scores;
  std::vector<int> candidate_class_ids;

  for (int row = 0; row < detections.rows; ++row) {
    const float * data = detections.ptr<float>(row);
    const float confidence = data[4];
    if (confidence < conf_threshold_) {
      continue;
    }

    int class_id = 0;
    if (detections.cols >= 6) {
      class_id = static_cast<int>(std::round(data[5]));
    }

    float x1 = data[0];
    float y1 = data[1];
    float x2 = data[2];
    float y2 = data[3];

    // Some exported models output normalized coordinates.
    if (x2 <= 1.5F && y2 <= 1.5F) {
      x1 *= static_cast<float>(input_size_);
      y1 *= static_cast<float>(input_size_);
      x2 *= static_cast<float>(input_size_);
      y2 *= static_cast<float>(input_size_);
    }

    x1 = (x1 - static_cast<float>(meta.pad_x)) / meta.scale;
    y1 = (y1 - static_cast<float>(meta.pad_y)) / meta.scale;
    x2 = (x2 - static_cast<float>(meta.pad_x)) / meta.scale;
    y2 = (y2 - static_cast<float>(meta.pad_y)) / meta.scale;

    x1 = ClipFloat(x1, 0.0F, static_cast<float>(original_size.width));
    y1 = ClipFloat(y1, 0.0F, static_cast<float>(original_size.height));
    x2 = ClipFloat(x2, 0.0F, static_cast<float>(original_size.width));
    y2 = ClipFloat(y2, 0.0F, static_cast<float>(original_size.height));

    cv::Rect box(
      static_cast<int>(std::round(x1)),
      static_cast<int>(std::round(y1)),
      static_cast<int>(std::round(std::max(0.0F, x2 - x1))),
      static_cast<int>(std::round(std::max(0.0F, y2 - y1))));
    box = ClipRect(box, original_size);
    if (box.area() <= 0) {
      continue;
    }

    candidate_boxes.push_back(box);
    candidate_scores.push_back(confidence);
    candidate_class_ids.push_back(class_id);
  }

  std::vector<int> kept_indices;
  cv::dnn::NMSBoxes(candidate_boxes, candidate_scores, conf_threshold_, nms_threshold_, kept_indices);

  std::vector<ObjectCandidate> results;
  results.reserve(kept_indices.size());
  for (int index : kept_indices) {
    ObjectCandidate candidate;
    candidate.class_id = candidate_class_ids[index];
    candidate.class_name = ResolveClassName(candidate.class_id);
    candidate.score = candidate_scores[index];
    candidate.bbox = candidate_boxes[index];
    results.push_back(std::move(candidate));
  }
  return results;
}

std::string YoloDetector::ResolveClassName(int class_id) const
{
  const auto it = class_map_.find(class_id);
  if (it != class_map_.end()) {
    return it->second;
  }
  return "class_" + std::to_string(class_id);
}

cv::Rect YoloDetector::ClipRect(const cv::Rect & box, const cv::Size & frame_size) const
{
  const int x = std::max(0, box.x);
  const int y = std::max(0, box.y);
  const int max_w = frame_size.width - x;
  const int max_h = frame_size.height - y;
  const int w = std::max(0, std::min(box.width, max_w));
  const int h = std::max(0, std::min(box.height, max_h));
  return cv::Rect(x, y, w, h);
}

}  // namespace vtol_vision
