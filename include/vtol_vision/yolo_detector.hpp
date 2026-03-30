#pragma once

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

#include <rclcpp/rclcpp.hpp>

#include <string>
#include <unordered_map>
#include <vector>

namespace vtol_vision
{

struct ObjectCandidate
{
  int class_id{0};
  std::string class_name;
  float score{0.0F};
  cv::Rect bbox;
};

class YoloDetector
{
public:
  explicit YoloDetector(rclcpp::Logger logger);

  static bool LoadClassMap(
    const std::string & yaml_path,
    std::unordered_map<int, std::string> & class_map,
    std::string & error_message);

  bool Initialize(
    const std::string & model_path,
    const std::unordered_map<int, std::string> & class_map,
    int input_size,
    float conf_threshold,
    float nms_threshold);

  std::vector<ObjectCandidate> Infer(const cv::Mat & frame);
  bool Ready() const {return is_ready_;}

private:
  struct LetterboxMeta
  {
    float scale{1.0F};
    int pad_x{0};
    int pad_y{0};
  };

  cv::Mat PrepareInput(const cv::Mat & frame, LetterboxMeta & meta) const;
  std::vector<ObjectCandidate> ParseCenterBoxOutput(
    const cv::Mat & detections,
    const LetterboxMeta & meta,
    const cv::Size & original_size) const;
  std::vector<ObjectCandidate> ParseCornerBoxOutput(
    const cv::Mat & detections,
    const LetterboxMeta & meta,
    const cv::Size & original_size) const;
  std::string ResolveClassName(int class_id) const;
  cv::Rect ClipRect(const cv::Rect & box, const cv::Size & frame_size) const;

  rclcpp::Logger logger_;
  cv::dnn::Net net_;
  bool is_ready_{false};
  int input_size_{640};
  float conf_threshold_{0.25F};
  float nms_threshold_{0.45F};
  std::unordered_map<int, std::string> class_map_;
};

}  // namespace vtol_vision

