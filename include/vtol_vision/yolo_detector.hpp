#pragma once

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>  // cv::dnn::NMSBoxes

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
  ~YoloDetector();

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

  // Minimal TensorRT logger — only surfaces errors and warnings.
  class TrtLogger : public nvinfer1::ILogger
  {
  public:
    void log(Severity severity, const char * msg) noexcept override;
  };

  cv::Mat PrepareInput(const cv::Mat & frame, LetterboxMeta & meta) const;

  // Parse output tensor [1, num_fields, num_candidates] (column-major over candidates).
  // Ultralytics single-class: num_fields=5 (cx,cy,w,h,score).
  // Multi-class: num_fields=4+N.
  std::vector<ObjectCandidate> PostProcess(
    const float * output,
    int num_candidates,
    int num_fields,
    const LetterboxMeta & meta,
    const cv::Size & original_size) const;

  std::string ResolveClassName(int class_id) const;
  cv::Rect ClipRect(const cv::Rect & box, const cv::Size & frame_size) const;

  rclcpp::Logger logger_;
  TrtLogger trt_logger_;

  nvinfer1::IRuntime * runtime_{nullptr};
  nvinfer1::ICudaEngine * engine_{nullptr};
  nvinfer1::IExecutionContext * context_{nullptr};

  void * gpu_input_{nullptr};
  void * gpu_output_{nullptr};
  std::vector<float> cpu_input_;
  std::vector<float> cpu_output_;
  cudaStream_t stream_{nullptr};

  // Determined at Initialize() from engine binding shapes.
  int num_candidates_{0};  // e.g. 8400 for 640-input YOLO
  int num_fields_{0};       // 4 + num_classes

  // Binding indices (TRT 8 enqueueV2).
  int input_binding_idx_{0};
  int output_binding_idx_{1};

  bool is_ready_{false};
  int input_size_{640};
  float conf_threshold_{0.25F};
  float nms_threshold_{0.45F};
  std::unordered_map<int, std::string> class_map_;
};

}  // namespace vtol_vision
