from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_share = FindPackageShare("vtol_vision")
    default_params_file = PathJoinSubstitution([package_share, "config", "vision_params.yaml"])
    default_class_map_file = PathJoinSubstitution([package_share, "config", "class_map.example.yaml"])

    params_file_arg = DeclareLaunchArgument(
        "params_file",
        default_value=default_params_file,
        description="ROS2 parameter file for vision_node",
    )
    camera_uri_arg = DeclareLaunchArgument(
        "camera_uri",
        default_value="0",
        description="camera index or stream path",
    )
    trt_engine_path_arg = DeclareLaunchArgument(
        "trt_engine_path",
        default_value="",
        description="ONNX/TensorRT model path used by YOLO backend",
    )
    class_map_yaml_arg = DeclareLaunchArgument(
        "class_map_yaml",
        default_value=default_class_map_file,
        description="class map yaml for YOLO class id to display name",
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

