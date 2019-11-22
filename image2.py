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

        self.joint_angles_pub = rospy.Publisher("/joint_angles", Float64MultiArray, queue_size = 10)

        self.target_pub_x = rospy.Publisher("/target_x", Float64, queue_size = 10)
        self.target_pub_y = rospy.Publisher("/target_y", Float64, queue_size = 10)
        self.target_pub_z = rospy.Publisher("/target_z", Float64, queue_size = 10)

        self.fk_x_pub = rospy.Publisher("/fk_x", Float64, queue_size = 10)
        self.fk_y_pub = rospy.Publisher("/fk_y", Float64, queue_size = 10)
        self.fk_z_pub = rospy.Publisher("/fk_z", Float64, queue_size = 10)

        self.joint1_pub = rospy.Publisher("/robot/joint1_position_controller/command", Float64, queue_size = 10)
        self.joint2_pub = rospy.Publisher("/robot/joint2_position_controller/command", Float64, queue_size = 10)
        self.joint3_pub = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size = 10)
        self.joint4_pub = rospy.Publisher("/robot/joint4_position_controller/command", Float64, queue_size = 10)

        # initialize a subscriber to recieve messages rom a topic named /robot/camera1/image_raw and use callback function to recieve data
        self.image_sub2 = rospy.Subscriber("/camera2/robot/image_raw",Image,self.callback2)
        # subscribe to the angles found in camera1
        self.camera1_angles = rospy.Subscriber("/camera1_positions",Float64MultiArray,self.callback3)


        self.image_sub1 = rospy.Subscriber("/chamfer",Float64MultiArray,self.target)

        # initialize the bridge between openCV and ROS

        self.time_previous_step = np.array([rospy.get_time()], dtype='float64')

        self.time_trajectory = rospy.get_time()

        self.q = [0,1,0,0]

        self.error = np.array([0,0,0], dtype = 'float64')
        self.error_d = np.array([0,0,0], dtype = 'float64')
        self.bridge = CvBridge()

#-----------------------------------Vision Part

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

    def detect_target(self, image, template):
        w, h = template.shape[::-1]
      	res = cv2.matchTemplate(image, template, 1)
    	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    	return np.array([min_loc[0] + w/2, max_loc[1]+h/2])

    def find_target(self, image ,z_y):
        mask = self.detect_orange(image)
        i = cv2.inRange(cv2.imread('image_crop.png', 1), (200, 200, 200), (255, 255, 255))

        z_x = self.detect_target(mask, i)
        orange3d = np.array([z_x[0], z_y[0], max(z_x[1], z_y[1])])
        self.orange3d = orange3d
        a = self.pixel2meter(image)
        orange3d = (orange3d * a)
        target = (orange3d - self.center3d)
        return target


    # Calculate the conversion from pixel to meter
    def pixel2meter(self,image):
        # Obtain the centre of 2 spheres
        circle1Pos = self.detect_yellow(image)
        circle2Pos = self.detect_blue(image)
        # find the distance between two circles
        dist = np.sum((circle2Pos - circle1Pos)**2)
        return 2 / np.sqrt(dist)

    # Calculate the relevant joint angles from the image
    def detect_joint_positions(self,image):
        a = self.pixel2meter(image)
        # Obtain the centre of each coloured blob
        center = a * self.detect_yellow(image)
        blue = a * self.detect_blue(image)
        b = (center- blue)

        green = a * self.detect_green(image)
        g = (center - green)

        red = a * self.detect_red(image)
        r = center - red

        return np.array([b[0],b[1], g[0], g[1], r[0], r[1], center[0],center[1]])

    def get_3d_points(self, camera1_positions):
        self.center3d = np.array([self.joints.data[6], camera1_positions[6], np.abs(max(self.joints.data[7], camera1_positions[7]))])
        self.yellow3d = np.array([0, 0, 0])
        self.blue3d = np.array( [0,0,2] )
        self.green3d = np.array( [self.joints.data[2], camera1_positions[2], np.abs(max(self.joints.data[3], camera1_positions[3]))] )
        self.red3d = np.array( [self.joints.data[4], camera1_positions[4], np.abs(max(self.joints.data[5], camera1_positions[5]))] )

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

    def f(self,theta, a, b):
        e = (self.r_x(theta).dot(a)) - b
        return sum(np.abs(e))

    def f1(self,theta, a, b):
        e = (self.r_y(theta).dot(a)) - b
        return sum(np.abs(e))

    def f2(self,theta, a, b):
        e = (self.r_z(theta).dot(a)) - b
        return sum(np.abs(e))

    def joint_angles(self,a, b):
        #b = b-a
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)

        l = least_squares(self.f, [0.5], args = (a,b), method = 'trf', bounds= (-np.pi/2, np.pi/2) )
        l1 = least_squares(self.f1, [0.5], args = (a,b), method = 'trf', bounds= (-np.pi/2, np.pi/2) )
        l2 = least_squares(self.f2, [0.5], args = (a,b), method = 'trf', bounds= (-np.pi/2, np.pi/2) )

        return np.array([l.x,l1.x,l2.x])

    def get_joint_angles(self, point1, point2, point3, point4):
        t123 = self.joint_angles( point3-point2, point2)
        t4 = self.joint_angles((point4-point3),(point3 - point2))
        #t4 = t4 - t123[0]
        t1234 = np.array([[0], t123[0], -t123[1], t4[0]])
        return np.reshape(np.round(t1234,2),4)

        #np.reshape(t1234,4)
#--------------------------------------------------Robot Control part

    def fk(self,q):
        t1 = q[0]
        t2 = q[1]
        t3 = q[2]
        t4 = q[3]

        p = np.pi/2

        x_e = 2.0*(np.sin(t3)*np.sin(t1 + p) + np.cos(t3)*np.cos(t1 + p)*np.cos(t2 + p))*np.cos(t4) + 3.0*np.sin(t3)*np.sin(t1 + p) - 2.0*np.sin(t4)*np.sin(t2 + p)*np.cos(t1 + p) + 3.0*np.cos(t3)*np.cos(t1 + p)*np.cos(t2 + p)
        y_e = -2.0*(np.sin(t3)*np.cos(t1 + p) - np.sin(t1 + p)*np.cos(t3)*np.cos(t2 + p))*np.cos(t4) - 3.0*np.sin(t3)*np.cos(t1 + p) - 2.0*np.sin(t4)*np.sin(t1 + p)*np.sin(t2 + p) + 3.0*np.sin(t1 + p)*np.cos(t3)*np.cos(t2 + p)
        z_e = 2.0*np.sin(t4)*np.cos(t2 + p) + 2.0*np.sin(t2 + p)*np.cos(t3)*np.cos(t4) + 3.0*np.sin(t2 + p)*np.cos(t3) + 2.0

        return np.array([x_e,y_e,z_e])

    def jacobian(self, q):
        t1 = q[0]
        t2 = q[1]
        t3 = q[2]
        t4 = q[3]

        j11 = (-2.0*np.sin(t1)*np.cos(t2)*np.cos(t3) + 2.0*np.sin(t3)*np.cos(t1))*np.cos(t4) + 2.0*np.sin(t1)*np.sin(t2)*np.sin(t4) - 3.0*np.sin(t1)*np.cos(t2)*np.cos(t3) + 3.0*np.sin(t3)*np.cos(t1)
        j12 = -2.0*np.sin(t2)*np.cos(t1)*np.cos(t3)*np.cos(t4) - 3.0*np.sin(t2)*np.cos(t1)*np.cos(t3) - 2.0*np.sin(t4)*np.cos(t1)*np.cos(t2)
        j13 = (2.0*np.sin(t1)*np.cos(t3) - 2.0*np.sin(t3)*np.cos(t1)*np.cos(t2))*np.cos(t4) + 3.0*np.sin(t1)*np.cos(t3) - 3.0*np.sin(t3)*np.cos(t1)*np.cos(t2)
        j14 = -(2.0*np.sin(t1)*np.sin(t3) + 2.0*np.cos(t1)*np.cos(t2)*np.cos(t3))*np.sin(t4) - 2.0*np.sin(t2)*np.cos(t1)*np.cos(t4)

        j21 = (2.0*np.sin(t1)*np.sin(t3) + 2.0*np.cos(t1)*np.cos(t2)*np.cos(t3))*np.cos(t4) + 3.0*np.sin(t1)*np.sin(t3) - 2.0*np.sin(t2)*np.sin(t4)*np.cos(t1) + 3.0*np.cos(t1)*np.cos(t2)*np.cos(t3)
        j22 = -2.0*np.sin(t1)*np.sin(t2)*np.cos(t3)*np.cos(t4) - 3.0*np.sin(t1)*np.sin(t2)*np.cos(t3) - 2.0*np.sin(t1)*np.sin(t4)*np.cos(t2)
        j23 = (-2.0*np.sin(t1)*np.sin(t3)*np.cos(t2) - 2.0*np.cos(t1)*np.cos(t3))*np.cos(t4) - 3.0*np.sin(t1)*np.sin(t3)*np.cos(t2) - 3.0*np.cos(t1)*np.cos(t3)
        j24 = -(2.0*np.sin(t1)*np.cos(t2)*np.cos(t3) - 2.0*np.sin(t3)*np.cos(t1))*np.sin(t4) - 2.0*np.sin(t1)*np.sin(t2)*np.cos(t4)

        j31 = 0
        j32 = -2.0*np.sin(t2)*np.sin(t4) + 2.0*np.cos(t2)*np.cos(t3)*np.cos(t4) + 3.0*np.cos(t2)*np.cos(t3)
        j33 = -2.0*np.sin(t2)*np.sin(t3)*np.cos(t4) - 3.0*np.sin(t2)*np.sin(t3)
        j34 = -2.0*np.sin(t2)*np.sin(t4)*np.cos(t3) + 2.0*np.cos(t2)*np.cos(t4)

        return np.array([[j11,j12,j13,j14],
                         [j21,j22,j23,j24],
                         [j31,j32,j33,j34]])



    def control(self,image):
        k_p = np.array([[0.5,0,0],
                        [0,0.5,0],
                        [0,0,0.5]])
        k_d = np.array([[0.005,0,0],
                       [0,0.005,0],
                       [0,0,0.005]])

        cur_time = np.array([rospy.get_time()])
        dt = cur_time - self.time_previous_step
        self.time_previous_step = cur_time

        pos = self.red3d

        pos_d = self.target

        self.error_d = ((pos_d - pos) - self.error) / dt
        self.error = pos_d - pos
        #print(self.get_joint_angles(self.yellow3d, self.blue3d, self.green3d, self.red3d))
        q = self.get_joint_angles(self.yellow3d, self.blue3d, self.green3d, self.red3d)
        #Pseudo inv
        j_inv = np.linalg.pinv(self.jacobian(q))
        dq_d = np.dot(j_inv, (np.dot(k_d, self.error_d.transpose()) + np.dot(k_p,self.error.transpose())))
        q_d = q + (dt * dq_d)

        return q_d

#---------------------------------Callback Functions

  # Recieve data, process it, and publish
    def callback2(self,data):
    # Recieve the image
        try:
            self.cv_image2 = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        # Uncomment if you want to save the image
        #cv2.imwrite('image_copy1.png', self.cv_image2)
        #im2=cv2.imshow('window2', self.cv_image2)
        cv2.waitKey(1)

        a = self.detect_joint_positions(self.cv_image2)
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

        self.get_3d_points(camera1_positions)

        #q = np.array([0.8,0.5,0.7,1.4])
        #self.fk(q)
        #print("x:", e[0], "y:", e[1], "z:", e[2])

        self.t1234 = Float64MultiArray()
        self.t1234.data = self.get_joint_angles(self.yellow3d, self.blue3d, self.green3d, self.red3d)

        try:
            self.joint_angles_pub.publish(self.t1234)
        except CvBridgeError as e:
            print(e)


    def target(self,data):
        try:
            z_y = data.data
        except CvBridgeError as e:
            print(e)

        self.target = self.find_target(self.cv_image2, z_y)
        #Uncomment for control
        '''
        q_d = self.control(self.cv_image2)
        #print(q_d)
        #q_d = np.mod(q_d, np.pi/2) * np.sign(q_d)

        #self.q = q_d
        #print(self.q)
        self.joint1 = Float64()
        self.joint1.data = q_d[0]


        self.joint2 = Float64()
        self.joint2.data = q_d[1]

        self.joint3 = Float64()
        self.joint3.data = q_d[2]

        self.joint4 = Float64()
        self.joint4.data = q_d[3]

        fk = self.fk(q_d)
        fk_x = Float64()
        fk_x.data = fk[0]

        fk_y = Float64()
        fk_y.data = fk[1]

        fk_z = Float64()
        fk_z.data = fk[2]
        '''
        try:
            #Uncomment for control
            '''
            self.fk_x_pub.publish(fk_x)
            self.fk_y_pub.publish(fk_y)
            self.fk_z_pub.publish(fk_z)

            self.joint1_pub.publish(self.joint1)
            self.joint2_pub.publish(self.joint2)
            self.joint3_pub.publish(self.joint3)
            self.joint4_pub.publish(self.joint4)
            '''
            self.target_pub_x.publish(self.target[0] )
            self.target_pub_y.publish(self.target[1] )
            self.target_pub_z.publish(np.abs(self.target[2]))
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
