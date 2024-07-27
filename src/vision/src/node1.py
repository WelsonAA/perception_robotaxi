#!/usr/bin/env python3
import time, rospy, cv2
from ros_numpy import numpify
from ultralytics import YOLO
from sensor_msgs.msg import Image
detection_model = YOLO("/home/welson03/Desktop/ever24/src/vision/src/yolov8s.pt")
rospy.init_node("ultralytics")
time.sleep(1)

def callback(data):
    """Callback function to process image and publish annotated images."""
    array = numpify(data)
    array = array[::-1]
    det_result = detection_model(array)
    det_annotated = det_result[0].plot(show=True)
    det_annotated_rgb = cv2.cvtColor(det_annotated, cv2.COLOR_BGR2RGB)
    # Display the image
    cv2.imshow('Image', det_annotated_rgb)
    cv2.waitKey(5)  

    #cv2.destroyAllWindows()

rospy.Subscriber("/camera/color/image_raw", Image, callback)

while True:
    rospy.spin()