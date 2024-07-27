#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from ultralytics import YOLO
model = YOLO('/home/welson03/Desktop/ever24/src/vision/src/yolov8s.pt')
class YoloV8ObjectTracker:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        self.depth_sub = rospy.Subscriber("/camera/depth/image_rect_raw", Image, self.depth_callback)
        self.rgb_kmatrix_sub = rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.rgb_kmatrix_callback)
        self.depth_kmatrix_sub = rospy.Subscriber("/camera/depth/camera_info", CameraInfo, self.depth_kmatrix_callback)
        self.detection_pub = rospy.Publisher("object_detection", String, queue_size=10)
        #self.model = YOLO('/home/welson03/Desktop/ever24/src/vision/src/yolov8s.pt')  # Update with your YOLOv8 model path

        self.depth_image = None
        self.rgb_k_matrix = None
        self.depth_k_matrix = None
        print("hiiii")
        

    def rgb_kmatrix_callback(self, data):
        self.rgb_k_matrix = np.array(data.K).reshape((3, 3))

    def depth_kmatrix_callback(self, data):
        self.depth_k_matrix = np.array(data.K).reshape((3, 3))

    def depth_callback(self, data):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(data, "passthrough")
            self.depth_image = cv2.flip(depth_image, 1)  # Flip depth image on the x-axis
        except CvBridgeError as e:
            print(e)
            self.depth_image = None

    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            cv_image = cv2.flip(cv_image, 1)  # Flip RGB image on the x-axis
        except CvBridgeError as e:
            print(e)
            return

        # Perform object detection and tracking
        results = model.track(cv_image)
        detected_objects = results.pandas().xyxy[0]

        if self.depth_image is not None and self.rgb_k_matrix is not None:
            for index, row in detected_objects.iterrows():
                xmin, ymin, xmax, ymax, confidence, class_id, name, track_id = row
                cv2.rectangle(cv_image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
                cv2.putText(cv_image, f"{name} ID:{track_id}", (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                # Calculate the center of the bounding box
                x_center = int((xmin + xmax) / 2)
                y_center = int((ymin + ymax) / 2)

                # Get the depth value at the center of the bounding box
                depth = self.depth_image[y_center, x_center]

                # Calculate 3D coordinates in the camera coordinate system using RGB camera matrix
                x = (x_center - self.rgb_k_matrix[0, 2]) * depth / self.rgb_k_matrix[0, 0]
                y = (y_center - self.rgb_k_matrix[1, 2]) * depth / self.rgb_k_matrix[1, 1]
                z = depth

                # Publish detected object with 3D coordinates and track ID
                self.detection_pub.publish(f"Detected {name} ID:{track_id} at ({x:.2f}, {y:.2f}, {z:.2f}) with confidence {confidence:.2f}")

        # Display the image with detections and tracking
        cv2.imshow("YOLOv8 Object Tracker", cv_image)
        cv2.waitKey(5)

def main():
    rospy.init_node('yolov8_object_tracker', anonymous=True)
    yolo_tracker = YoloV8ObjectTracker()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()