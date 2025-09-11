from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='image_bridge',
            executable='bridge_node', 
            name='image_segmentation_node' 
            ),
    ])
