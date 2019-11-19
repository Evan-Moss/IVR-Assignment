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
from scipy.optimize import least_squares

class image_converter:

    # Defines publisher and subscriber
    def __init__(self):
        # initialize the node named image_processing
        rospy.init_node('image_processing', anonymous=True)
        # initialize a publisher to send images from camera2 to a topic named image_topic2
        self.image_pub2 = rospy.Publisher("image_topic2",Image, queue_size = 1)

        self.euler_pub = rospy.Publisher("/euler", Float64MultiArray, queue_size = 10)

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


    def detect_orange(self, image):
        mask = cv2.inRange(image, (50, 100, 110), (90, 185, 220))
        return mask




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

    def r_x(self,theta):
        return np.array([[1,0,0],
                         [0,np.cos(theta), -np.sin(theta)],
                         [0,np.sin(theta), np.cos(theta)]])

    def r_y(self,theta):
        return np.array([[np.cos(theta),0,-np.sin(theta)],
                         [0,1,0],
                         [np.sin(theta), 0, np.cos(theta)]])

    def r_z(self,theta):
        return np.array([[np.cos(theta),-np.sin(theta),0],
                         [np.sin(theta), np.cos(theta),0],
                         [0,0,1]])

    def r(self,theta_x,theta_y,theta_z):
        return np.dot( self.r_z(theta_z), self.r_y(theta_y).dot(self.r_x(theta_x)) )

    def f(self,theta, a, b):
        e = (self.r_x(theta).dot(a)) - b
        return sum(np.abs(e))

    def f1(self,theta, a, b):
        e = (self.r_y(theta).dot(a)) - b
        return sum(np.abs(e))

    def f2(self,theta, a, b):
        e = (self.r_z(theta).dot(a)) - b
        return sum(np.abs(e))

    def euler(self,r):
        psi = np.arctan2(r[2][1], r[2][2])
        theta = np.arctan2(-r[2][0], np.sqrt( (r[2][1] ** 2) + (r[2][2] ** 2) ))
        phi = np.arctan2(r[1][0], r[0][0])
        return np.array([psi, theta, phi])

    def get_joint_angles(self,a, b):
        b = b-a
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)

        l = least_squares(self.f, [0.5], args = (a,b), method = 'lm' )
        l1 = least_squares(self.f1, [0.5], args = (a,b), method = 'lm' )
        l2 = least_squares(self.f2, [0.5], args = (a,b), method = 'lm' )

        r1 = self.r(l.x,l1.x,l2.x)

        return self.euler(r1)




  # Recieve data, process it, and publish
    def callback2(self,data):
    # Recieve the image
        try:
            self.cv_image2 = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        # Uncomment if you want to save the image
        #cv2.imwrite('image_copy.png', cv_image)
        #im2=cv2.imshow('window2', self.cv_image2)
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


        yellow3d = np.array([0, 0, 0])
        blue3d = np.array( [0,0,2] )
        green3d = np.array( [self.joints.data[2], camera1_positions[2], max(self.joints.data[3], camera1_positions[3])] )
        red3d = np.array( [self.joints.data[4], camera1_positions[4], max(self.joints.data[5], camera1_positions[5])] )
        #print(yellow3d, blue3d, green3d, red3d)

        #print("y:", yellow3d)
        #print("b:",blue3d)
        #print("g:",green3d)
        #print("r:",red3d)

        eu = self.get_joint_angles(blue3d, green3d)

        self.eu = Float64MultiArray()
        self.eu.data = eu
        #print("x:", e[0], "y:", e[1], "z:", e[2])
        try:
            self.euler_pub.publish(self.eu)
        except CvBridgeError as e:
            print(e)


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
