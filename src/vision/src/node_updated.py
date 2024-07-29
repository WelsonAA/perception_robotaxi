#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from ultralytics import YOLO

class YoloV8ObjectTracker:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        self.depth_sub = rospy.Subscriber("/camera/depth/image_rect_raw", Image, self.depth_callback)
        self.rgb_kmatrix_sub = rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.rgb_kmatrix_callback)
        self.depth_kmatrix_sub = rospy.Subscriber("/camera/depth/camera_info", CameraInfo, self.depth_kmatrix_callback)
        self.detection_pub = rospy.Publisher("object_detection", String, queue_size=10)
        self.model = YOLO('/home/welson03/Desktop/ever24/src/vision/src/best.pt')  # Update with your YOLOv8 model path

        self.depth_image = None
        self.rgb_k_matrix = None
        self.depth_k_matrix = None
        self.min_depth = 0.2  # Minimum depth in meters
        self.max_depth = 3.0  # Maximum depth in meters

    def rgb_kmatrix_callback(self, data):
        self.rgb_k_matrix = np.array(data.K).reshape((3, 3))

    def depth_kmatrix_callback(self, data):
        self.depth_k_matrix = np.array(data.K).reshape((3, 3))

    def depth_callback(self, data):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(data, "passthrough")
            self.depth_image = cv2.rotate(depth_image, cv2.ROTATE_180)
            #self.depth_image = depth_image[:, ::-1]  # Flip depth image on the x-axis using NumPy slicing
            #rospy.loginfo("Depth image flipped")
        except CvBridgeError as e:
            rospy.logerr(f"Failed to convert depth image: {e}")
            self.depth_image = None

    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            cv_image = cv2.rotate(cv_image, cv2.ROTATE_180)
            #cv_image = cv_image[:, ::-1]  # Flip RGB image on the x-axis using NumPy slicing
            #rospy.loginfo("RGB image flipped")
        except CvBridgeError as e:
            rospy.logerr(f"Failed to convert RGB image: {e}")
            return
        
        # Verify image type and shape
        rospy.loginfo(f"cv_image type: {type(cv_image)}, shape: {cv_image.shape}")

        # Perform object detection
        results = self.model(cv_image)

        if self.depth_image is not None and self.rgb_k_matrix is not None:
            for result in results:
                boxes = result.boxes  # Get the boxes attribute
                for box in boxes:
                    # Extract coordinates
                    xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    name = result.names[class_id]

                    try:
                        # Draw the bounding box
                        cv2.rectangle(cv_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                        # Calculate the center of the bounding box
                        x_center = (xmin + xmax) // 2
                        y_center = (ymin + ymax) // 2

                        # Get the depth value at the center of the bounding box
                        depth_value = self.depth_image[y_center, x_center]

                        # Convert grayscale depth value to meters
                        depth = self.min_depth + (self.max_depth - self.min_depth) * (255 - depth_value) / 255.0

                        # Calculate 3D coordinates in the camera coordinate system using RGB camera matrix
                        x = (x_center - self.rgb_k_matrix[0, 2]) * depth / self.rgb_k_matrix[0, 0]
                        y = (y_center - self.rgb_k_matrix[1, 2]) * depth / self.rgb_k_matrix[1, 1]
                        z = depth

                        # Create the label with name, confidence, and coordinates
                        label = f"{name} {confidence:.2f} ({xmin},{ymin})"
                        coordinates = f"({x:.2f}, {y:.2f}, {z:.2f})"
                        cv2.putText(cv_image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)
                        cv2.putText(cv_image, coordinates, (xmin, ymin + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)

                        # Publish detected object with 3D coordinates
                        self.detection_pub.publish(f"Detected {name} at ({x:.2f}, {y:.2f}, {z:.2f}) with confidence {confidence:.2f}")

                    except Exception as e:
                        rospy.logerr(f"Failed to draw rectangle or text on cv_image: {e}")
                        continue

        # Display the image with detections
        cv2.imshow("YOLOv8 Object Tracker", cv_image)
        cv2.waitKey(3)

def main():
    rospy.init_node('yolov8_object_tracker', anonymous=True)
    yolo_tracker = YoloV8ObjectTracker()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
