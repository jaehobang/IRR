#!/usr/bin/env python

import rospy
import cv2
import sys
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Point
from geometry_msgs.msg import Twist



class DriveWheels:
  def __init__(self):
    rospy.init_node('ballFinder', anonymous = True)
    rospy.Subscriber("/turtlebot3/burger/ball_location", Point, self.callback)
    self.vel_publisher = rospy.Publisher("/cmd_vel", Twist, queue_size = 1)
    self.publish_rate = rospy.Rate(10)  # 10hz

    self.center_points = []
    self.P = 0.1

    rospy.spin()


  def callback(self, data):
    center_x = data.z / 2
    self.center_points.append(data)

    if len(self.center_points) < 2:
      previous_point = Point()
      previous_point.x = data.z / 2
    else:
      previous_point = self.center_points[-2]
    current_point = self.center_points[-1]
    x_offset = (current_point.x - center_x) #if positive, need to move to right
    #TODO: Need to debug this step and see if applying to yaw is the correct way to go
    if x_offset > 50:
      movement_vel = -0.1
    elif x_offset < -50:
      movement_vel = 0.1
    else:
      movement_vel = 0
    twist = Twist()
    twist.angular.z = movement_vel

    self.vel_publisher.publish(twist)
    self.publish_rate.sleep()




if __name__=="__main__":
  DriveWheels()

