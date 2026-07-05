#include "vtol_vision/vision_node.hpp"

#include <cv_bridge/cv_bridge.h>
#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace vtol_vision
{

namespace
{
std::unordered_map<int, std::string> DefaultClassMap()
{
  return {
    {0, "basket"},
    {1, "mannequin"}
  };
}

geometry_msgs::msg::Quaternion RotationMatrixToQuaternion(const cv::Mat & rotation_matrix)
{
  geometry_msgs::msg::Quaternion q;

  const double m00 = rotation_matrix.at<double>(0, 0);
  const double m11 = rotation_matrix.at<double>(1, 1);
  const double m22 = rotation_matrix.at<double>(2, 2);
  const double trace = m00 + m11 + m22;

  if (trace > 0.0) {
    const double s = std::sqrt(trace + 1.0) * 2.0;
    q.w = 0.25 * s;
    q.x = (rotation_matrix.at<double>(2, 1) - rotation_matrix.at<double>(1, 2)) / s;
    q.y = (rotation_matrix.at<double>(0, 2) - rotation_matrix.at<double>(2, 0)) / s;
    q.z = (rotation_matrix.at<double>(1, 0) - rotation_matrix.at<double>(0, 1)) / s;
  } else if (m00 > m11 && m00 > m22) {
    const double s = std::sqrt(1.0 + m00 - m11 - m22) * 2.0;
    q.w = (rotation_matrix.at<double>(2, 1) - rotation_matrix.at<double>(1, 2)) / s;
    q.x = 0.25 * s;
    q.y = (rotation_matrix.at<double>(0, 1) + rotation_matrix.at<double>(1, 0)) / s;
    q.z = (rotation_matrix.at<double>(0, 2) + rotation_matrix.at<double>(2, 0)) / s;
  } else if (m11 > m22) {
    const double s = std::sqrt(1.0 + m11 - m00 - m22) * 2.0;
    q.w = (rotation_matrix.at<double>(0, 2) - rotation_matrix.at<double>(2, 0)) / s;
    q.x = (rotation_matrix.at<double>(0, 1) + rotation_matrix.at<double>(1, 0)) / s;
    q.y = 0.25 * s;
    q.z = (rotation_matrix.at<double>(1, 2) + rotation_matrix.at<double>(2, 1)) / s;
  } else {
    const double s = std::sqrt(1.0 + m22 - m00 - m11) * 2.0;
    q.w = (rotation_matrix.at<double>(1, 0) - rotation_matrix.at<double>(0, 1)) / s;
    q.x = (rotation_matrix.at<double>(0, 2) + rotation_matrix.at<double>(2, 0)) / s;
    q.y = (rotation_matrix.at<double>(1, 2) + rotation_matrix.at<double>(2, 1)) / s;
    q.z = 0.25 * s;
  }
  return q;
}

}  // namespace

VisionNode::VisionNode()
: Node("vision_node")
{
  LoadParameters();

  aruco_pub_ = this->create_publisher<msg::VisionDetections>("/vision/aruco", pub_queue_depth_);
  objects_pub_ = this->create_publisher<msg::VisionDetections>("/vision/objects", pub_queue_depth_);
  if (enable_debug_image_) {
    debug_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
      "/vision/debug_image",
      pub_queue_depth_);
  }

  aruco_dictionary_ = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_50);
  aruco_parameters_ = cv::aruco::DetectorParameters::create();

  // --- 고도 40m 소형 마커 탐지 튜닝 ---
  // 작은 겉보기 크기를 허용: 기본값 0.03 → 0.02
  aruco_parameters_->minMarkerPerimeterRate = aruco_min_perimeter_rate_;
  // 넓은 adaptive threshold 윈도우 범위 → 작은 셀도 이진화 가능
  aruco_parameters_->adaptiveThreshWinSizeMin = 3;
  aruco_parameters_->adaptiveThreshWinSizeMax = 53;
  aruco_parameters_->adaptiveThreshWinSizeStep = 10;
  // 원거리 투시 왜곡 허용 범위 약간 완화
  aruco_parameters_->polygonalApproxAccuracyRate = 0.05;
  // 서브픽셀 코너 정제 → 포즈 추정 정확도 향상
  aruco_parameters_->cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
  aruco_parameters_->cornerRefinementWinSize = 5;
  aruco_parameters_->cornerRefinementMaxIterations = 30;
  aruco_parameters_->cornerRefinementMinAccuracy = 0.1;

  if (!LoadCameraCalibration(camera_info_yaml_)) {
    RCLCPP_WARN(
      this->get_logger(),
      "camera calibration is unavailable. ArUco pose/reprojection quality can degrade.");
  }

  if (!OpenCamera(camera_uri_)) {
    throw std::runtime_error("Failed to open camera_uri: " + camera_uri_);
  }
  capture_.set(cv::CAP_PROP_FRAME_WIDTH, static_cast<double>(frame_width_));
  capture_.set(cv::CAP_PROP_FRAME_HEIGHT, static_cast<double>(frame_height_));
  capture_.set(cv::CAP_PROP_FPS, static_cast<double>(camera_fps_));

  std::unordered_map<int, std::string> class_map;
  std::string class_map_error;
  if (!class_map_yaml_.empty()) {
    if (!YoloDetector::LoadClassMap(class_map_yaml_, class_map, class_map_error)) {
      RCLCPP_WARN(
        this->get_logger(),
        "failed to load class map (%s). fallback to defaults.",
        class_map_error.c_str());
      class_map = DefaultClassMap();
    }
  } else {
    class_map = DefaultClassMap();
  }

  yolo_detector_ = std::make_unique<YoloDetector>(this->get_logger());
  if (!yolo_detector_->Initialize(
      trt_engine_path_,
      class_map,
      yolo_input_size_,
      conf_thr_,
      nms_thr_))
  {
    RCLCPP_WARN(
      this->get_logger(),
      "YOLO is disabled until trt_engine_path points to a readable model.");
  } else {
    RCLCPP_INFO(
      this->get_logger(),
      "YOLO ready: model=%s input=%d conf=%.2f nms=%.2f class_map=%s",
      trt_engine_path_.c_str(),
      yolo_input_size_,
      static_cast<double>(conf_thr_),
      static_cast<double>(nms_thr_),
      class_map_yaml_.empty() ? "<default>" : class_map_yaml_.c_str());
  }

  running_.store(true);
  capture_thread_ = std::thread(&VisionNode::CaptureLoop, this);
  yolo_thread_ = std::thread(&VisionNode::YoloLoop, this);

  RCLCPP_INFO(this->get_logger(), "vision_node started.");
}

VisionNode::~VisionNode()
{
  running_.store(false);
  if (capture_thread_.joinable()) {
    capture_thread_.join();
  }
  if (yolo_thread_.joinable()) {
    yolo_thread_.join();
  }
  if (capture_.isOpened()) {
    capture_.release();
  }
}

void VisionNode::LoadParameters()
{
  camera_uri_ = this->declare_parameter<std::string>("camera_uri", "0");
  camera_info_yaml_ = this->declare_parameter<std::string>("camera_info_yaml", "");
  frame_id_ = this->declare_parameter<std::string>("frame_id", "camera");
  marker_size_m_ = this->declare_parameter<double>("marker_size_m", 0.15);
  aruco_allowed_ids_param_ = this->declare_parameter<std::vector<int64_t>>("aruco_allowed_ids", {});
  trt_engine_path_ = this->declare_parameter<std::string>("trt_engine_path", "");
  class_map_yaml_ = this->declare_parameter<std::string>("class_map_yaml", "");
  conf_thr_ = static_cast<float>(this->declare_parameter<double>("conf_thr", 0.25));
  nms_thr_ = static_cast<float>(this->declare_parameter<double>("nms_thr", 0.45));
  aruco_min_perimeter_rate_ = this->declare_parameter<double>("aruco_min_perimeter_rate", 0.02);
  yolo_input_size_ = this->declare_parameter<int>("yolo_input_size", 640);
  frame_width_ = this->declare_parameter<int>("frame_width", 640);
  frame_height_ = this->declare_parameter<int>("frame_height", 480);
  camera_fps_ = this->declare_parameter<int>("camera_fps", 30);
  yolo_period_ms_ = this->declare_parameter<int>("yolo_period_ms", 50);
  yolo_debug_log_period_ms_ = this->declare_parameter<int>("yolo_debug_log_period_ms", 2000);
  pub_queue_depth_ = this->declare_parameter<int>("queue_size", 10);
  enable_debug_image_ = this->declare_parameter<bool>("enable_debug_image", false);
  enable_yolo_debug_log_ = this->declare_parameter<bool>("enable_yolo_debug_log", true);
  undistort_image_ = this->declare_parameter<bool>("undistort_image", true);

  allowed_ids_.clear();
  for (const int64_t id : aruco_allowed_ids_param_) {
    if (id >= 0) {
      allowed_ids_.insert(static_cast<int>(id));
    }
  }
}

bool VisionNode::OpenCamera(const std::string & camera_uri)
{
  char * end_ptr = nullptr;
  const long maybe_index = std::strtol(camera_uri.c_str(), &end_ptr, 10);
  if (end_ptr != nullptr && *end_ptr == '\0') {
    return capture_.open(static_cast<int>(maybe_index), cv::CAP_ANY);
  }
  return capture_.open(camera_uri, cv::CAP_ANY);
}

bool VisionNode::LoadCameraCalibration(const std::string & yaml_path)
{
  has_calibration_ = false;
  camera_matrix_ = cv::Mat::eye(3, 3, CV_64F);
  dist_coeffs_ = cv::Mat::zeros(1, 5, CV_64F);

  if (yaml_path.empty()) {
    return false;
  }

  cv::FileStorage fs(yaml_path, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    RCLCPP_WARN(this->get_logger(), "failed to open camera_info_yaml: %s", yaml_path.c_str());
    return false;
  }

  auto parse_matrix_node = [](const cv::FileNode & node) -> cv::Mat {
      if (node.empty()) {
        return {};
      }
      if (node.isMap() && !node["data"].empty()) {
        std::vector<double> data;
        node["data"] >> data;
        const int rows = static_cast<int>(node["rows"]);
        const int cols = static_cast<int>(node["cols"]);
        if (rows > 0 && cols > 0 && static_cast<int>(data.size()) == rows * cols) {
          return cv::Mat(rows, cols, CV_64F, data.data()).clone();
        }
      }
      cv::Mat mat;
      node >> mat;
      return mat;
    };

  cv::Mat camera_matrix = parse_matrix_node(fs["camera_matrix"]);
  cv::Mat dist_coeffs = parse_matrix_node(fs["distortion_coefficients"]);
  if (dist_coeffs.empty()) {
    dist_coeffs = parse_matrix_node(fs["dist_coeff"]);
  }
  if (dist_coeffs.empty()) {
    dist_coeffs = parse_matrix_node(fs["distCoeffs"]);
  }

  if (camera_matrix.empty()) {
    RCLCPP_WARN(this->get_logger(), "camera matrix is missing in %s", yaml_path.c_str());
    return false;
  }

  camera_matrix_ = camera_matrix.clone();
  if (!dist_coeffs.empty()) {
    dist_coeffs_ = dist_coeffs.reshape(1, 1).clone();
  }
  has_calibration_ = true;
  return true;
}

void VisionNode::CaptureLoop()
{
  while (rclcpp::ok() && running_.load()) {
    cv::Mat raw_frame;
    if (!capture_.read(raw_frame)) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(),
        *this->get_clock(),
        2000,
        "camera frame read failed");
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
      continue;
    }

    const rclcpp::Time capture_stamp = this->now();
    cv::Mat frame = raw_frame;
    if (undistort_image_ && has_calibration_) {
      cv::undistort(raw_frame, frame, camera_matrix_, dist_coeffs_);
    }

    std_msgs::msg::Header header;
    header.stamp = capture_stamp;
    header.frame_id = frame_id_;

    const ArucoResult aruco_result = DetectAruco(frame, header);
    PublishArucoDetections(header, aruco_result.detections, capture_stamp);

    if (enable_debug_image_ && debug_image_pub_) {
      cv::Mat debug_frame = frame.clone();
      if (!aruco_result.ids.empty()) {
        cv::aruco::drawDetectedMarkers(debug_frame, aruco_result.corners, aruco_result.ids);
      }
      PublishDebugImage(debug_frame, header);
    }

    {
      std::lock_guard<std::mutex> lock(frame_mutex_);
      latest_frame_.image = frame.clone();
      latest_frame_.stamp = capture_stamp;
      latest_frame_.sequence += 1;
      has_latest_frame_ = true;
    }
  }
}

void VisionNode::YoloLoop()
{
  uint64_t last_sequence = 0;

  while (rclcpp::ok() && running_.load()) {
    const auto loop_start = std::chrono::steady_clock::now();

    if (!yolo_detector_ || !yolo_detector_->Ready()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(300));
      continue;
    }

    SharedFrame frame_copy;
    bool has_frame = false;
    {
      std::lock_guard<std::mutex> lock(frame_mutex_);
      has_frame = has_latest_frame_;
      if (has_frame) {
        frame_copy = latest_frame_;
      }
    }

    if (!has_frame || frame_copy.sequence == 0 || frame_copy.sequence == last_sequence) {
      std::this_thread::sleep_for(std::chrono::milliseconds(3));
      continue;
    }

    last_sequence = frame_copy.sequence;
    const std::vector<ObjectCandidate> detections = yolo_detector_->Infer(frame_copy.image);

    std_msgs::msg::Header header;
    header.stamp = frame_copy.stamp;
    header.frame_id = frame_id_;
    PublishObjectDetections(header, detections, frame_copy.stamp);

    if (enable_yolo_debug_log_) {
      float best_score = 0.0F;
      for (const auto & detection : detections) {
        best_score = std::max(best_score, detection.score);
      }

      std::ostringstream summary;
      summary << "YOLO detections=" << detections.size();
      if (!detections.empty()) {
        summary << " best=" << detections.front().class_name
                << " score=" << best_score;
      }
      summary << " frame=" << frame_copy.image.cols << "x" << frame_copy.image.rows;

      RCLCPP_INFO_THROTTLE(
        this->get_logger(),
        *this->get_clock(),
        yolo_debug_log_period_ms_,
        "%s",
        summary.str().c_str());
    }

    if (yolo_period_ms_ > 0) {
      const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - loop_start);
      const int remaining_ms = yolo_period_ms_ - static_cast<int>(elapsed.count());
      if (remaining_ms > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(remaining_ms));
      }
    }
  }
}

ArucoResult VisionNode::DetectAruco(const cv::Mat & frame, const std_msgs::msg::Header & header) const
{
  ArucoResult result;

  std::vector<int> ids;
  std::vector<std::vector<cv::Point2f>> corners;
  std::vector<std::vector<cv::Point2f>> rejected;
  cv::aruco::detectMarkers(frame, aruco_dictionary_, corners, ids, aruco_parameters_, rejected);

  if (ids.empty()) {
    return result;
  }

  std::vector<int> filtered_ids;
  std::vector<std::vector<cv::Point2f>> filtered_corners;
  filtered_ids.reserve(ids.size());
  filtered_corners.reserve(corners.size());

  for (size_t i = 0; i < ids.size(); ++i) {
    if (!allowed_ids_.empty() && allowed_ids_.count(ids[i]) == 0U) {
      continue;
    }
    filtered_ids.push_back(ids[i]);
    filtered_corners.push_back(corners[i]);
  }

  if (filtered_ids.empty()) {
    return result;
  }

  result.ids = filtered_ids;
  result.corners = filtered_corners;

  std::vector<cv::Vec3d> rvecs(filtered_ids.size(), cv::Vec3d(0.0, 0.0, 0.0));
  std::vector<cv::Vec3d> tvecs(filtered_ids.size(), cv::Vec3d(0.0, 0.0, 0.0));

  if (has_calibration_) {
    cv::aruco::estimatePoseSingleMarkers(
      filtered_corners,
      static_cast<float>(marker_size_m_),
      camera_matrix_,
      dist_coeffs_,
      rvecs,
      tvecs);
  }

  result.detections.reserve(filtered_ids.size());
  for (size_t i = 0; i < filtered_ids.size(); ++i) {
    msg::ArucoDetection detection;
    detection.header = header;
    detection.marker_id = filtered_ids[i];

    if (has_calibration_) {
      detection.pose = BuildPoseFromRt(rvecs[i], tvecs[i]);
      detection.reprojection_error = ComputeReprojectionError(filtered_corners[i], rvecs[i], tvecs[i]);
      detection.confidence = 1.0F / (1.0F + detection.reprojection_error);
    } else {
      detection.pose.orientation.w = 1.0;
      detection.reprojection_error = -1.0F;
      detection.confidence = 0.0F;
    }

    result.detections.push_back(std::move(detection));
  }
  return result;
}

geometry_msgs::msg::Pose VisionNode::BuildPoseFromRt(const cv::Vec3d & rvec, const cv::Vec3d & tvec) const
{
  cv::Mat rotation_matrix;
  cv::Rodrigues(rvec, rotation_matrix);

  geometry_msgs::msg::Pose pose;
  pose.position.x = tvec[0];
  pose.position.y = tvec[1];
  pose.position.z = tvec[2];
  pose.orientation = RotationMatrixToQuaternion(rotation_matrix);
  return pose;
}

float VisionNode::ComputeReprojectionError(
  const std::vector<cv::Point2f> & image_points,
  const cv::Vec3d & rvec,
  const cv::Vec3d & tvec) const
{
  if (!has_calibration_ || image_points.size() != 4U) {
    return -1.0F;
  }

  const float half_size = static_cast<float>(marker_size_m_ * 0.5);
  const std::vector<cv::Point3f> object_points = {
    {-half_size, half_size, 0.0F},
    {half_size, half_size, 0.0F},
    {half_size, -half_size, 0.0F},
    {-half_size, -half_size, 0.0F}
  };

  std::vector<cv::Point2f> projected_points;
  cv::projectPoints(object_points, rvec, tvec, camera_matrix_, dist_coeffs_, projected_points);
  if (projected_points.size() != image_points.size()) {
    return -1.0F;
  }

  double total_error = 0.0;
  for (size_t i = 0; i < image_points.size(); ++i) {
    total_error += cv::norm(projected_points[i] - image_points[i]);
  }
  return static_cast<float>(total_error / static_cast<double>(image_points.size()));
}

void VisionNode::PublishArucoDetections(
  const std_msgs::msg::Header & header,
  const std::vector<msg::ArucoDetection> & detections,
  const rclcpp::Time & capture_stamp)
{
  msg::VisionDetections packet;
  packet.header = header;
  packet.aruco_detections = detections;
  packet.pipeline_latency_ms = static_cast<float>((this->now() - capture_stamp).seconds() * 1000.0);
  aruco_pub_->publish(std::move(packet));
}

void VisionNode::PublishObjectDetections(
  const std_msgs::msg::Header & header,
  const std::vector<ObjectCandidate> & detections,
  const rclcpp::Time & capture_stamp)
{
  msg::VisionDetections packet;
  packet.header = header;
  packet.object_detections.reserve(detections.size());
  for (const auto & candidate : detections) {
    msg::ObjectDetection det;
    det.header = header;
    det.class_id = candidate.class_id;
    det.class_name = candidate.class_name;
    det.score = candidate.score;
    det.bbox.x_offset = static_cast<uint32_t>(std::max(0, candidate.bbox.x));
    det.bbox.y_offset = static_cast<uint32_t>(std::max(0, candidate.bbox.y));
    det.bbox.width = static_cast<uint32_t>(std::max(0, candidate.bbox.width));
    det.bbox.height = static_cast<uint32_t>(std::max(0, candidate.bbox.height));
    det.bbox.do_rectify = false;
    packet.object_detections.push_back(std::move(det));
  }
  packet.pipeline_latency_ms = static_cast<float>((this->now() - capture_stamp).seconds() * 1000.0);
  objects_pub_->publish(std::move(packet));
}

void VisionNode::PublishDebugImage(const cv::Mat & frame, const std_msgs::msg::Header & header) const
{
  if (!debug_image_pub_) {
    return;
  }
  auto msg = cv_bridge::CvImage(header, "bgr8", frame).toImageMsg();
  debug_image_pub_->publish(*msg);
}

}  // namespace vtol_vision
