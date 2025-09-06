import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image as PILImage
import torchvision.transforms as transforms
import os
from ament_index_python.packages import get_package_share_directory

class ImageProcessingNode(Node):
    def __init__(self):
        super().__init__('image_processing_node')
        
        self.declare_parameter('input_topic', '/camera/image_raw')
        self.declare_parameter('output_topic', '/inference/image_overlay')
        self.declare_parameter('onnx_provider', 'CUDAExecutionProvider')

        input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        onnx_provider = self.get_parameter('onnx_provider').get_parameter_value().string_value
        
        self.bridge = CvBridge()
        
        package_share_directory = get_package_share_directory('image_bridge')
        onnx_model_path = os.path.join(package_share_directory, 'resource', 'ddrnet23.onnx')

        self.subscription = self.create_subscription(
            Image,
            input_topic,
            self.image_callback,
            10)
        
        self.publisher = self.create_publisher(Image, output_topic, 10)
        
        self.get_logger().info(f"Loading ONNX model from: {onnx_model_path}")
        try:
            self.onnx_session = ort.InferenceSession(
                onnx_model_path, 
                providers=[onnx_provider]
            )
            self.get_logger().info(f"Model loaded successfully. Using provider: {self.onnx_session.get_providers()[0]}")
        except Exception as e:
            self.get_logger().error(f"Error loading ONNX model: {e}")
            raise
            
        self.transform = transforms.Compose([
            transforms.Resize((180, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.palette = np.zeros((256, 3), dtype=np.uint8)
        self.palette[1] = [0, 255, 0] 

        self.get_logger().info(f"Node '{self.get_name()}' started successfully.")
        self.get_logger().info(f"Subscribing to: {input_topic}")
        self.get_logger().info(f"Publishing to: {output_topic}")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'Could not convert image: {e}')
            return

        image_pil = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        input_tensor = self.transform(image_pil).unsqueeze(0)

        ort_inputs = {'inputx': input_tensor.numpy()}
        ort_outs = self.onnx_session.run(['outputy'], ort_inputs)
        prediction = ort_outs[0][0]

        color_mask_rgb = self.palette[prediction.astype(np.uint8)]
        color_mask_resized = cv2.resize(
            color_mask_rgb, 
            (cv_image.shape[1], cv_image.shape[0]), 
            interpolation=cv2.INTER_NEAREST
        )
        color_mask_bgr = cv2.cvtColor(color_mask_resized, cv2.COLOR_RGB2BGR)
        overlay_image = cv2.addWeighted(cv_image, 0.6, color_mask_bgr, 0.4, 0)

        try:
            overlay_msg = self.bridge.cv2_to_imgmsg(overlay_image, "bgr8")
            overlay_msg.header = msg.header
            self.publisher.publish(overlay_msg)
        except Exception as e:
            self.get_logger().error(f'Could not publish image: {e}')


def main(args=None):
    rclpy.init(args=args)
    image_processing_node = ImageProcessingNode()
    try:
        rclpy.spin(image_processing_node)
    except KeyboardInterrupt:
        pass
    finally:
        image_processing_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
