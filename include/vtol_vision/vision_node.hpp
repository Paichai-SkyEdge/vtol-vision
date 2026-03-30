#pragma once

#include "vtol_vision/msg/aruco_detection.hpp"
#include "vtol_vision/msg/object_detection.hpp"
#include "vtol_vision/msg/vision_detections.hpp"
#include "vtol_vision/yolo_detector.hpp"

#include <opencv2/aruco.hpp>
#include <opencv2/opencv.hpp>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

namespace vtol_vision
{

struct ArucoResult
{
  std::vector<int> ids;
  std::vector<std::vector<cv::Point2f>> corners;
  std::vector<msg::ArucoDetection> detections;
};

class VisionNode : public rclcpp::Node
{
public:
  VisionNode();
  ~VisionNode() override;

private:
  struct SharedFrame
  {
    cv::Mat image;
    rclcpp::Time stamp;
    uint64_t sequence{0};
  };

  void LoadParameters();
  bool OpenCamera(const std::string & camera_uri);
  bool LoadCameraCalibration(const std::string & yaml_path);

  void CaptureLoop();
  void YoloLoop();

  ArucoResult DetectAruco(const cv::Mat & frame, const std_msgs::msg::Header & header) const;
  geometry_msgs::msg::Pose BuildPoseFromRt(const cv::Vec3d & rvec, const cv::Vec3d & tvec) const;
  float ComputeReprojectionError(
    const std::vector<cv::Point2f> & image_points,
    const cv::Vec3d & rvec,
    const cv::Vec3d & tvec) const;

  void PublishArucoDetections(
    const std_msgs::msg::Header & header,
    const std::vector<msg::ArucoDetection> & detections,
    const rclcpp::Time & capture_stamp);
  void PublishObjectDetections(
    const std_msgs::msg::Header & header,
    const std::vector<ObjectCandidate> & detections,
    const rclcpp::Time & capture_stamp);
  void PublishDebugImage(const cv::Mat & frame, const std_msgs::msg::Header & header) const;

  // Parameters
  std::string camera_uri_;
  std::string camera_info_yaml_;
  std::string frame_id_;
  double marker_size_m_{0.15};
  std::vector<int64_t> aruco_allowed_ids_param_;
  std::unordered_set<int> allowed_ids_;
  std::string trt_engine_path_;
  std::string class_map_yaml_;
  float conf_thr_{0.25F};
  float nms_thr_{0.45F};
  int yolo_input_size_{640};
  int frame_width_{640};
  int frame_height_{480};
  int camera_fps_{30};
  int yolo_period_ms_{50};
  int pub_queue_depth_{10};
  bool enable_debug_image_{false};
  bool undistort_image_{true};

  // Runtime state
  std::atomic<bool> running_{false};
  mutable std::mutex frame_mutex_;
  SharedFrame latest_frame_;
  bool has_latest_frame_{false};
  bool has_calibration_{false};

  cv::VideoCapture capture_;
  cv::Mat camera_matrix_;
  cv::Mat dist_coeffs_;
  cv::Ptr<cv::aruco::Dictionary> aruco_dictionary_;
  cv::Ptr<cv::aruco::DetectorParameters> aruco_parameters_;
  std::unique_ptr<YoloDetector> yolo_detector_;

  std::thread capture_thread_;
  std::thread yolo_thread_;

  rclcpp::Publisher<msg::VisionDetections>::SharedPtr aruco_pub_;
  rclcpp::Publisher<msg::VisionDetections>::SharedPtr objects_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_image_pub_;
};

}  // namespace vtol_vision
