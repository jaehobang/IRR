#!/usr/bin/env python
# license removed for brevity
import rospy
import cv2
import sys
import numpy as np
import math
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Point
from sensor_msgs.msg import LaserScan

class getObjectRange:

    def __init__(self, debug_mode):
        rospy.init_node('getObjectRange', anonymous=True)
        rospy.Subscriber("/turtlebot3/burger/ball_location", Point, self.callback_image)
        rospy.Subscribe("/scan", LaserScan, self.callback_laser)

        # TODO: Figure out the angle with some math formulation
        self.actual_ball_radius = 0.15 #Everything needs to be in meters
        self.assumed_ball_range = [0.5, 1.5] #Give a bound of angles for rigor

        self.calculated_angle_range = []
        self.calculated_ball_offset = -1



        self.loc_publisher = rospy.Publisher("/turtlebot3/burger/ball_range", Point, queue_size=1)
        self.publish_rate = rospy.Rate(10)  # 10hz

        self.debug_mode = debug_mode
        if debug_mode:
            self.window_name = "image"
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.startWindowThread()

        rospy.spin()

        if debug_mode:
            cv2.destroyAllWindows()

    #pixel_offset refers to distance of object from the center of image in terms of pixels
    #pixel radius refers to radius of object in the image in terms of pixels
    #return values will be in radians
    def calculate_angle_range(self, pixel_offset, pixel_radius):
        x2 = pixel_offset
        x1 = pixel_radius
        y1 = self.actual_ball_radius
        y2 = x2 * y1 / x1
        self.calculated_ball_offset = y2
        return [math.atan2(y2, self.assumed_ball_range[1]), math.atan2(y2, self.assumed_ball_range[0])]

    #will return a list of ranges because it might be sliced
    def convertRad2Index(self, estimated_rad_range, length_of_array, min_rad, max_rad, rad_increment):
        #Note the LaserScan topic starts from 0 rad to pi rad
        #0 rad refers to where the robot is facing, and value of rad increases as it goes cc-wise
        #TODO: look at the orientation of the sensors and apply the appropriate transformation
        pass


    def callback_laser(self, data):
        if self.calculated_angle_range == []:
            return

        index_ranges = self.convertRad2Index(self.calculated_angle_range, len(data.ranges),
                                            data.angle_min, data.angle_max, data.angle_increment)

        minimum_dists = []
        for index_range in index_ranges:
            minimum_dists.append( min(data.ranges[index_range[0]:index_range[1]]) )

        ball_dist = min(minimum_dists)

        pub_point = Point()
        pub_point.x = ball_dist
        pub_point.y = self.calculated_ball_offset
        pub_point.z = math.atan2(pub_point.y, pub_point.x) #calculated in radians

        self.loc_publisher.publish(pub_point)
        self.publish_rate.sleep()
        #TODO: publish that result to drive_wheels


    def callback_image(self, data):
        # Retrieve the image, find the location, publish that
        # rospy.logerr(type(data.format))
        # rospy.logerr(data.format)
        # rospy.logerr(type(data.data))
        # rospy.logerr(data.data)
        # tmp = []
        # assert(type(compressedImage) == type(tmp))
        # assert(type(compressedImage) == type("hello world"))

        image_width = data.z
        center_x = image_width / 2

        current_point = data.x
        current_radius = data.y

        x_offset = (current_point.x - center_x)  # if positive, need to move to right
        self.calculated_angle_range = self.calculate_angle_range(x_offset, current_radius)

if __name__ == '__main__':
  args = sys.argv
  if len(args)  == 2 and args[1] == 'debug':
    getObjectRange(True)
  else:
    getObjectRange(False)