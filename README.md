!!! Main Node (bridge_node)
/opt/ros_venv/bin/python /root/project_ws/src/image_bridge/image_bridge/bridge_node.py --ros-args -p input_topic:=/camera/camera1/color/image_raw

!!! Inference Overlay (relay_node)
LIBGL_ALWAYS_SOFTWARE=1 ros2 run image_bridge relay_node

