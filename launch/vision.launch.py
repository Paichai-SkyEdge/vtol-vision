from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_share = FindPackageShare("vtol_vision")
    default_params_file = PathJoinSubstitution([package_share, "config", "vision_params.yaml"])
    default_class_map_file = PathJoinSubstitution([package_share, "config", "class_map.example.yaml"])
    # Engine path is overridden at runtime on the target device.
    # Generate best.engine on the Jetson with:
    # yolo export model=weights/mannequin_yolo11n/best.pt format=engine device=0 imgsz=640 half=True
    default_engine_path = ""

    params_file_arg = DeclareLaunchArgument(
        "params_file",
        default_value=default_params_file,
        description="ROS2 parameter YAML for vision_node",
    )
    camera_uri_arg = DeclareLaunchArgument(
        "camera_uri",
        default_value="0",
        description=(
            "Camera index (e.g. '0') or GStreamer pipeline string for CSI cameras. "
            "Example CSI pipeline: "
            "'nvarguscamerasrc ! video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1 "
            "! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! appsink'"
        ),
    )
    trt_engine_path_arg = DeclareLaunchArgument(
        "trt_engine_path",
        default_value=default_engine_path,
        description=(
            "Absolute path to a TensorRT .engine file generated on this device. "
            "Must match the Jetson GPU architecture — cannot be transferred from another machine."
        ),
    )
    class_map_yaml_arg = DeclareLaunchArgument(
        "class_map_yaml",
        default_value=default_class_map_file,
        description="YAML mapping class id → display name",
    )

    vision_node = Node(
        package="vtol_vision",
        executable="vision_node",
        name="vision_node",
        output="screen",
        parameters=[
            LaunchConfiguration("params_file"),
            {
                "camera_uri": LaunchConfiguration("camera_uri"),
                "trt_engine_path": LaunchConfiguration("trt_engine_path"),
                "class_map_yaml": LaunchConfiguration("class_map_yaml"),
            },
        ],
    )

    return LaunchDescription(
        [
            params_file_arg,
            camera_uri_arg,
            trt_engine_path_arg,
            class_map_yaml_arg,
            vision_node,
        ]
    )
