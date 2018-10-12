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
        rospy.init_node('getObjectRange', anonymous=True, log_level=rospy.DEBUG)
        rospy.Subscriber("/scan", LaserScan, self.callback_laser, queue_size = 1)

        # TODO: Figure out the angle with some math formulation

    #pixel_offset refers to distance of object from the center of image in terms of pixels
    #pixel radius refers to radius of object in the image in terms of pixels
    #return values will be in radians
    def calculate_angle_range(self, pixel_offset, pixel_radius):
        x2 = pixel_offset
        x1 = pixel_radius
        y1 = self.actual_ball_radius
        y2 = x2 * y1 / x1
        self.calculated_ball_offset = y2
        if pixel_offset > 0:
            self.calculated_angle_range = [2*math.pi - math.atan2(y2, self.assumed_ball_range[0]),                                     2*math.pi - math.atan2(y2, self.assumed_ball_range[1])]
        else:
            self.calculated_angle_range = [math.atan2(math.fabs(y2), self.assumed_ball_range[1]),
                                           math.atan2(math.fabs(y2), self.assumed_ball_range[0])]
        return

    #will return a list of ranges because it might be sliced
    def convertRad2Index(self, length_of_array, min_rad, max_rad, rad_increment):
        #Note the LaserScan topic starts from 0 rad to pi rad
        #0 rad refers to where the robot is facing, and value of rad increases as it goes cc-wise
        #TODO: look at the orientation of the sensors and apply the appropriate transformation
        start_index = int(self.calculated_angle_range[0] / rad_increment)
        end_index = int(self.calculated_angle_range[1] / rad_increment)
        return [start_index, end_index]

    def callback_laser(self, data):
        print len(data.ranges)
        print type(data.ranges)
        '''
        index_ranges = self.convertRad2Index(len(data.ranges),
                                            data.angle_min, data.angle_max, data.angle_increment)

        #rospy.loginfo("index ranges are " + str(index_ranges))

        minimum_dist = np.array(data.ranges[index_ranges[0]:index_ranges[1]])
        minimum_dist[minimum_dist < data.range_min] = math.inf

        ball_dist = min(minimum_dist)

        pub_point = Point()
        pub_point.x = ball_dist
        pub_point.y = self.calculated_ball_offset
        pub_point.z = math.atan2(pub_point.y, pub_point.x) #calculated in radians

        rospy.loginfo("x,y,z: " + str(pub_point.x) + "," + str(pub_point.y) + "," + str(pub_point.z))
        self.loc_publisher.publish(pub_point)
        self.publish_rate.sleep()
        '''



if __name__ == '__main__':
  args = sys.argv
  if len(args)  == 2 and args[1] == 'debug':
    getObjectRange(True)
  else:
    getObjectRange(False)
