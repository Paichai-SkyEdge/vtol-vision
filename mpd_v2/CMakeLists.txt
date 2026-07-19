cmake_minimum_required(VERSION 3.16)
project(realsense_live_detect LANGUAGES CXX C CXX)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
else()
  set(WITH_TENSORRT OFF)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio dnn highgui calib3d)

option(WITH_REALSENSE "Enable RealSense D435i support" ON)
option(WITH_TENSORRT   "Enable TensorRT backend (Jetson)" OFF)

# --- RealSense ---
if(WITH_REALSENSE)
  find_package(realsense2 QUIET)
  if(realsense2_FOUND)
    add_compile_definitions(HAS_REALSENSE)
    message(STATUS "RealSense: ON")
  else()
    message(STATUS "RealSense: NOT FOUND (sim mode)")
  endif()
endif()

# --- TensorRT ---
if(WITH_TENSORRT AND CMAKE_CUDA_COMPILER)
  find_package(CUDA REQUIRED)
  set(TRT_INC "" CACHE PATH "TensorRT include")
  set(TRT_LIB "" CACHE PATH "TensorRT lib")
  if(NOT TRT_INC)
    foreach(p /usr/include/aarch64-linux-gnu /usr/include/x86_64-linux-gnu)
      if(EXISTS "${p}/NvInfer.h")
        set(TRT_INC "${p}")
        break()
      endif()
    endforeach()
  endif()
  if(NOT TRT_LIB)
    foreach(p /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu)
      if(EXISTS "${p}/libnvinfer.so")
        set(TRT_LIB "${p}")
        break()
      endif()
    endforeach()
  endif()
  if(TRT_INC AND TRT_LIB)
    add_compile_definitions(HAS_TENSORRT)
    message(STATUS "TensorRT: ON  inc=${TRT_INC} lib=${TRT_LIB}")
  else()
    message(STATUS "TensorRT: NOT FOUND (DNN/ONNX fallback)")
  endif()
endif()

# --- YOLO Detector Library ---
add_library(yolo_detector STATIC
  src/yolo_detector.cpp
)
target_include_directories(yolo_detector PUBLIC
  ${OpenCV_INCLUDE_DIRS}
  include
)
target_link_libraries(yolo_detector PUBLIC
  ${OpenCV_LIBS}
)
if(WITH_TENSORRT AND CMAKE_CUDA_COMPILER AND TRT_INC AND TRT_LIB)
  target_include_directories(yolo_detector PUBLIC ${TRT_INC})
  target_link_libraries(yolo_detector PUBLIC
    ${CUDA_LIBRARIES}
    ${CUDA_CUDART_LIBRARY}
    ${TRT_LIB}/libnvinfer.so
    ${TRT_LIB}/libnvinfer_plugin.so
    dl
  )
endif()
target_compile_definitions(yolo_detector PRIVATE $<$<CONFIG:Debug>:DEBUG>)

# --- Main App ---
add_executable(realsense_live_detect
  src/realsense_live_detect.cpp
)
target_include_directories(realsense_live_detect PRIVATE
  ${OpenCV_INCLUDE_DIRS}
  include
)
target_link_libraries(realsense_live_detect
  yolo_detector
  ${OpenCV_LIBS}
  pthread
)
if(WITH_REALSENSE AND realsense2_FOUND)
  target_include_directories(realsense_live_detect PRIVATE ${realsense2_INCLUDE_DIR})
  target_link_libraries(realsense_live_detect ${realsense2_LIBRARY})
endif()
target_compile_definitions(realsense_live_detect PRIVATE $<$<CONFIG:Debug>:DEBUG>)

install(TARGETS realsense_live_detect DESTINATION bin)
install(DIRECTORY config/  DESTINATION share/realsense_live_detect/config)
install(DIRECTORY weights/ DESTINATION share/realsense_live_detect/weights)
install(PROGRAMS scripts/export_engine.sh scripts/run.sh scripts/run_sim.sh
        DESTINATION share/realsense_live_detect/scripts)
