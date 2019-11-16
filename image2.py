#!/usr/bin/env python
#Z-X plane
import roslib
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Float64
from cv_bridge import CvBridge, CvBridgeError


class image_converter:

    # Defines publisher and subscriber
    def __init__(self):
        # initialize the node named image_processing
        rospy.init_node('image_processing', anonymous=True)
        # initialize a publisher to send images from camera2 to a topic named image_topic2
        self.image_pub2 = rospy.Publisher("image_topic2",Image, queue_size = 1)
        # initialize a subscriber to recieve messages rom a topic named /robot/camera1/image_raw and use callback function to recieve data
        self.image_sub2 = rospy.Subscriber("/camera2/robot/image_raw",Image,self.callback2)
        # subscribe to the angles found in camera1
        self.camera1_angles = rospy.Subscriber("/camera1_positions",Float64MultiArray,self.callback3)
        # initialize the bridge between openCV and ROS
        self.bridge = CvBridge()

    def detect_red(self,image):
        # Isolate the blue colour in the image as a binary image
        mask = cv2.inRange(image, (0, 0, 100), (0, 0, 255))
        # This applies a dilate that makes the binary region larger (the more iterations the larger it becomes)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
        # Obtain the moments of the binary image
        M = cv2.moments(mask)
        # Calculate pixel coordinates for the centre of the blob
        if M['m00'] == 0:
            cx = 0
            cy = 0
            return np.array([cx, cy])
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        return np.array([cx, cy])

    def detect_green(self,image):
        mask = cv2.inRange(image, (0, 100, 0), (0, 255, 0))
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
        M = cv2.moments(mask)
        if M['m00'] == 0:
            cx = 0
            cy = 0
            return np.array([cx, cy])
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        return np.array([cx, cy])

    # Detecting the centre of the blue circle
    def detect_blue(self,image):
        mask = cv2.inRange(image, (100, 0, 0), (255, 0, 0))
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
        M = cv2.moments(mask)
        if M['m00'] == 0:
            cx = 0
            cy = 0
            return np.array([cx, cy])
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        return np.array([cx, cy])

    # Detecting the centre of the yellow circle
    def detect_yellow(self,image):
        mask = cv2.inRange(image, (0, 100, 100), (0, 255, 255))
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
        M = cv2.moments(mask)
        if M['m00'] == 0:
            cx = 0
            cy = 0
            return np.array([cx, cy])
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        return np.array([cx, cy])

    # Calculate the conversion from pixel to meter
    def pixel2meter(self,image):
        # Obtain the centre of each coloured blob
        circle1Pos = self.detect_yellow(image)
        circle2Pos = self.detect_blue(image)
        # find the distance between two circles
        dist = np.sum((circle1Pos - circle2Pos)**2)
        return 2 / np.sqrt(dist)

    # Calculate the relevant joint angles from the image
    def detect_joint_angles(self,image):
        a = self.pixel2meter(image)
        # Obtain the centre of each coloured blob
        center = a * self.detect_yellow(image)

        blue = a * self.detect_blue(image)
        b = (center- blue)
        #print("distance from yellow to blue:", y2b)

        green = a * self.detect_green(image)
        g = (center - green)
        #rint("distance from yellow to green:", y2g)

        red = a * self.detect_red(image)
        r = (center - red)
        #print("distance from yellow to red:", y2r)

        return np.array([b[0],b[1], g[0], g[1], r[0], r[1]])

  # Recieve data, process it, and publish
    def callback2(self,data):
    # Recieve the image
        try:
            self.cv_image2 = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        # Uncomment if you want to save the image
        #cv2.imwrite('image_copy.png', cv_image)
        im2=cv2.imshow('window2', self.cv_image2)
        cv2.waitKey(1)

        a = self.detect_joint_angles(self.cv_image2)
        self.joints = Float64MultiArray()
        self.joints.data = a

        # Publish the results
        try:
            self.image_pub2.publish(self.bridge.cv2_to_imgmsg(self.cv_image2, "bgr8"))
        except CvBridgeError as e:
            print(e)



    # Recieve data, process it, and publish
    def callback3(self,data):
    # Recieve the image
        try:
            camera1_positions = data.data
            #print(camera1_positions)
            #print("Camera 2:", self.joints.data)
        except CvBridgeError as e:
            print(e)

        def angle(a, b):
            dot = np.dot(a,b)
            magab = np.linalg.norm(a) * np.linalg.norm(b)
            if(magab == 0):
                return 0.0
            return np.arccos( dot / magab )

        yellow3d = np.array([0, 0, 0])
        blue3d = np.array([self.joints.data[0], camera1_positions[0], (self.joints.data[1] + camera1_positions[1])/2])
        green3d = np.array([self.joints.data[2], camera1_positions[2], (self.joints.data[3] + camera1_positions[3])/2])
        red3d = np.array([self.joints.data[4], camera1_positions[4], (self.joints.data[5] + camera1_positions[5])/2])
        print(yellow3d, blue3d, green3d, red3d)
        yb_ang = angle(yellow3d, blue3d)
        bg_ang = angle(blue3d, green3d)
        gr_ang = np.arctan2(np.cross(green3d,red3d), np.dot(green3d,red3d))
        print("Yellow to Blue:", yb_ang)
        print("Blue to Green:",bg_ang)
        print("Green to Red:",gr_ang)
# call the class
def main(args):
    ic = image_converter()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

# run the code if the node is called
if __name__ == '__main__':
    main(sys.argv)
