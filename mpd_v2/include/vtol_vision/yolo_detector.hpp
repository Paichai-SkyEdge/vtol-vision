#pragma once

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

#include <string>
#include <unordered_map>
#include <vector>

#ifdef HAS_TENSORRT
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#endif

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
  explicit YoloDetector();
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

  enum class Backend { DNN, TensorRT };
  Backend backend() const { return backend_; }

private:
  struct LetterboxMeta
  {
    float scale{1.0F};
    int pad_x{0};
    int pad_y{0};
  };

  cv::Mat PrepareInputDNN(const cv::Mat & frame, LetterboxMeta & meta) const;
  cv::Mat PrepareInputTRT(const cv::Mat & frame, LetterboxMeta & meta) const;

  std::vector<ObjectCandidate> PostProcessDNN(
    const cv::Mat & output,
    const LetterboxMeta & meta,
    const cv::Size & original_size) const;

  std::vector<ObjectCandidate> PostProcessTRT(
    const float * output,
    int num_candidates,
    int num_fields,
    const LetterboxMeta & meta,
    const cv::Size & original_size) const;

  std::string ResolveClassName(int class_id) const;
  cv::Rect ClipRect(const cv::Rect & box, const cv::Size & frame_size) const;

  // --- DNN (ONNX) backend ---
  bool InitDNN(const std::string & model_path);
  std::vector<ObjectCandidate> InferDNN(const cv::Mat & frame);

  // --- TensorRT (.engine) backend ---
  bool InitTRT(const std::string & model_path);
  std::vector<ObjectCandidate> InferTRT(const cv::Mat & frame);

  Backend backend_{Backend::DNN};
  int input_size_{640};
  float conf_threshold_{0.25F};
  float nms_threshold_{0.45F};
  std::unordered_map<int, std::string> class_map_;
  bool is_ready_{false};

  // DNN
  cv::dnn::Net dnn_net_;

  // TensorRT
#ifdef HAS_TENSORRT
  class TrtLogger : public nvinfer1::ILogger
  {
  public:
    void log(Severity severity, const char * msg) noexcept override;
  };
  TrtLogger trt_logger_;
  nvinfer1::IRuntime * runtime_{nullptr};
  nvinfer1::ICudaEngine * engine_{nullptr};
  nvinfer1::IExecutionContext * context_{nullptr};
  void * gpu_input_{nullptr};
  void * gpu_output_{nullptr};
  std::vector<float> cpu_input_;
  std::vector<float> cpu_output_;
  cudaStream_t stream_{nullptr};
  int num_candidates_{0};
  int num_fields_{0};
  int input_binding_idx_{0};
  int output_binding_idx_{1};
#endif
};

}  // namespace vtol_vision
