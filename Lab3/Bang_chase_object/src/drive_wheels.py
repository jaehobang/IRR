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
    rospy.Subscriber("/turtlebot3/burger/ball_range", Point, self.callback)
    self.vel_publisher = rospy.Publisher("/cmd_vel", Twist, queue_size = 1)
    self.publish_rate = rospy.Rate(1)  # 10hz

    self.center_points = []
    self.P = 0.1
    self.optimal_dist = 1 # 1 meter

    rospy.spin()


  def callback(self, data):
    length_offset = data.x
    width_offset = data.y
    angle_offset = data.z

    if data.x == 0 and data.y == 0 and data.z == 0:
      movement_vel = 0.3
      twist = Twist()
      twist.angular.z = movement_vel
      self.vel_publisher.publish(twist)
      self.publish_rate.sleep()
    else:

      if angle_offset > 0.5:
        movement_vel = -0.1
      elif angle_offset < -0.5:
        movement_vel = 0.1
      else:
        movement_vel = 0
      twist = Twist()
      twist.angular.z = movement_vel

      self.vel_publisher.publish(twist)
      self.publish_rate.sleep()

      twist = Twist()
      twist.linear.x = 0.1 * (length_offset - self.optimal_dist)

      self.vel_publisher.publish(twist)
      self.publish_rate.sleep()





if __name__=="__main__":
  DriveWheels()

