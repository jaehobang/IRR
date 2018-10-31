#!/usr/bin/env python
# license removed for brevity
import rospy
import sys
import numpy as np
import math
from geometry_msgs.msg import Point
from sensor_msgs.msg import LaserScan



# This class is going to take in an estimated
class getObjectRange:
    def __init__(self, debug_mode):
        rospy.init_node('getObjectRange', anonymous=True, log_level=rospy.DEBUG)
        rospy.Subscriber("/scan", LaserScan, self.callback_laser, queue_size = 1)

        # TODO: Figure out the angle with some math formulation
        self.loc_publisher = rospy.Publisher("/turtlebot3/emergency_obstacle", Point, queue_size = 1)
        self.emergency_dist = 0.5
        self.publish_rate = rospy.Rate(10)
        rospy.spin()

    def callback_laser(self, data):
        length = len(data.ranges)
        left_limit = length / 12 #we will look at 0 - 30 degrees and 330 - 360 degrees
        points_of_interest = np.array(data.ranges[:left_limit][::-1] + data.ranges[-left_limit:][::-1])

        points_of_interest[points_of_interest < data.range_min] = float('inf')
        ball_dist = min(points_of_interest)

        if ball_dist < self.emergency_dist:

            pub_point = Point()
            pub_point.x = ball_dist
            pub_point.y = np.where(points_of_interest == ball_dist)[0][0] * 1.04 / len(points_of_interest) #60 degrees is 1.04 rad
            rospy.loginfo("obs_location x: " + str(pub_point.x) + "m\t"  + "y: " + str(math.degrees(pub_point.y)) + " degrees\t")
            self.loc_publisher.publish(pub_point)
            self.publish_rate.sleep()



if __name__ == '__main__':
  args = sys.argv
  if len(args)  == 2 and args[1] == 'debug':
    getObjectRange(True)
  else:
    getObjectRange(False)
