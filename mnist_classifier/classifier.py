import os

import cv2
import torch
import torch.nn as nn
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from cv_bridge import CvBridge

from ament_index_python.packages import get_package_share_directory
# ------------------------------------------------
# CNN model
# ------------------------------------------------




class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Linear(10 * 12 * 12, 10)

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


# ------------------------------------------------
# ROS 2 Node
# ------------------------------------------------

class MNISTClassifierNode(Node):

    def __init__(self):
        super().__init__("mnist_classifier")

        # Select GPU if available
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.get_logger().info(
            f"Using device: {self.device}"
        )

        # ----------------------------------------
        # Load neural network
        # ----------------------------------------

        self.model = SimpleCNN().to(self.device)

        # model_path = os.path.join(
        #     os.path.dirname(__file__),
        #     "simple_cnn_mnist.pth"
        # )

        package_share = get_package_share_directory(
            "mnist_classifier"
        )

        model_path = os.path.join(
            package_share,
            "models",
            "simple_cnn_mnist.pth"
        )

        self.model.load_state_dict(
            torch.load(
                model_path,
                map_location=self.device
            )
        )

        self.model.eval()

        self.get_logger().info(
            f"Loaded model: {model_path}"
        )

        # ----------------------------------------
        # ROS image conversion
        # ----------------------------------------

        self.bridge = CvBridge()

        # ----------------------------------------
        # Subscriber
        # ----------------------------------------

        self.subscription = self.create_subscription(
            Image,
            "/camera/image_raw",
            self.image_callback,
            10
        )

        # ----------------------------------------
        # Prediction publisher
        # ----------------------------------------

        self.prediction_publisher = self.create_publisher(
            Int32,
            "/mnist_prediction",
            10
        )

        self.get_logger().info(
            "MNIST classifier node started"
        )

        self.get_logger().info(
            "Listening to /camera/image_raw"
        )

        self.get_logger().info(
            "Publishing predictions on /mnist_prediction"
        )


    # ------------------------------------------------
    # Image preprocessing
    # ------------------------------------------------

    def preprocess_image(self, cv_image):

        # Convert RGB/BGR camera image to grayscale
        if len(cv_image.shape) == 3:
            gray = cv2.cvtColor(
                cv_image,
                cv2.COLOR_BGR2GRAY
            )
        else:
            gray = cv_image

        # Resize to MNIST dimensions
        gray = cv2.resize(
            gray,
            (28, 28)
        )

        # ------------------------------------------------
        # Optional threshold
        #
        # MNIST consists roughly of:
        # white digit
        # black background
        #
        # Depending on your camera, inversion may be
        # necessary.
        # ------------------------------------------------

        _, gray = cv2.threshold(
            gray,
            127,
            255,
            cv2.THRESH_BINARY
        )

        # Uncomment this if the physical image contains
        # a black digit on white paper.
        #
        # gray = cv2.bitwise_not(gray)

        # Convert [0,255] -> [0,1]
        image = gray.astype(np.float32) / 255.0

        # MNIST normalization
        image = (image - 0.1307) / 0.3081

        # Convert NumPy -> Tensor
        image_tensor = torch.from_numpy(image)

        # Shape:
        #
        # [28,28]
        # ->
        # [1,1,28,28]
        #
        # batch x channel x height x width

        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)

        return image_tensor.to(self.device)


    # ------------------------------------------------
    # ROS image callback
    # ------------------------------------------------

    def image_callback(self, msg):

        try:

            self.get_logger().info(
                "Received image"
            )
            # ROS Image -> OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8"
            )

            # Preprocess
            image_tensor = self.preprocess_image(
                cv_image
            )

            # Neural network inference
            with torch.no_grad():

                output = self.model(
                    image_tensor
                )

                probabilities = torch.softmax(
                    output,
                    dim=1
                )

                prediction = torch.argmax(
                    probabilities,
                    dim=1
                ).item()

                confidence = probabilities[
                    0,
                    prediction
                ].item()

            # Print prediction
            self.get_logger().info(
                f"Prediction: {prediction}, "
                f"confidence: {confidence:.3f}"
            )

            # Publish prediction
            prediction_msg = Int32()

            prediction_msg.data = prediction

            self.prediction_publisher.publish(
                prediction_msg
            )

        except Exception as error:

            self.get_logger().error(
                f"Image processing error: {error}"
            )


# ------------------------------------------------
# Main
# ------------------------------------------------

def main(args=None):

    rclpy.init(args=args)

    node = MNISTClassifierNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()