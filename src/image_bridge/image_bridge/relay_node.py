import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class RelayNode(Node):
    def __init__(self):
        super().__init__('relay_node')
        
        self.bridge = CvBridge()
        self.window_name = "Inference Overlay"
        
        self.subscription = self.create_subscription(
            Image,
            '/inference/image_overlay',
            self.image_callback,
            10)
        
        self.get_logger().info("Relay node started, waiting for image...")
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)

    def image_callback(self, msg):

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'Could not convert image: {e}')
            return
            
        cv2.imshow(self.window_name, cv_image)
        cv2.waitKey(1)

    def destroy_node(self):

        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    relay_node = RelayNode()
    try:
        rclpy.spin(relay_node)
    except KeyboardInterrupt:
        pass
    finally:
        relay_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
