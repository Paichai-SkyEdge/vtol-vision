cmake_minimum_required(VERSION 3.8)
project(vtol_vision)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio dnn aruco calib3d)
find_package(rclcpp REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)

# --- CUDA ---
find_package(CUDA REQUIRED)

# --- TensorRT ---
# Install: sudo apt install tensorrt libnvinfer-dev libnvinfer-plugin-dev
find_library(NVINFER_LIB nvinfer
  HINTS
    /usr/lib/aarch64-linux-gnu
    /usr/lib/x86_64-linux-gnu
    /usr/local/lib
  REQUIRED)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS
    /usr/include/aarch64-linux-gnu
    /usr/include/x86_64-linux-gnu
    /usr/local/include
  REQUIRED)
message(STATUS "TensorRT: ${NVINFER_LIB}  include: ${TENSORRT_INCLUDE_DIR}")

set(msg_files
  "msg/ArucoDetection.msg"
  "msg/ObjectDetection.msg"
  "msg/VisionDetections.msg"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  DEPENDENCIES std_msgs geometry_msgs sensor_msgs
)

add_library(yolo_detector src/yolo_detector.cpp)
target_include_directories(yolo_detector
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
)
ament_target_dependencies(yolo_detector rclcpp)
target_link_libraries(yolo_detector
  ${OpenCV_LIBRARIES}
  ${NVINFER_LIB}
  ${CUDA_LIBRARIES}
)

add_executable(vision_node
  src/main.cpp
  src/vision_node.cpp
)
target_include_directories(vision_node
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)
ament_target_dependencies(vision_node
  cv_bridge
  geometry_msgs
  rclcpp
  sensor_msgs
  std_msgs
)
target_link_libraries(vision_node yolo_detector ${OpenCV_LIBRARIES})

add_dependencies(vision_node ${PROJECT_NAME}__rosidl_generator_cpp)
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(vision_node ${cpp_typesupport_target})

install(TARGETS
  vision_node
  yolo_detector
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_export_dependencies(rosidl_default_runtime)
ament_export_include_directories(include)
ament_package()
