#!/usr/bin/env python
import math
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
    rospy.init_node('driveWheels', anonymous = True, log_level=rospy.DEBUG)
    rospy.Subscriber("/turtlebot3/burger/ball_range", Point, self.callback, queue_size = 1)
    self.vel_publisher = rospy.Publisher("/cmd_vel", Twist, queue_size = 10)
    self.publish_rate = rospy.Rate(10)  # 10hz

    self.center_points = []
    self.P = 0.1
    self.optimal_dist = 0.4 # 1 meter

    rospy.spin()


  def callback(self, data):
    length_offset = data.x
    width_offset = data.y
    angle_error = data.z
    rospy.loginfo(str(data.x) + "," + str(data.y) + "," + str(data.z))
    p_value_linear = 0.1
    p_value_angular = 0.1

    if data.x == 0 and data.y == 0 and data.z == 0:
      rospy.loginfo("No detection for 1 sec")

      movement_vel = 0.5
      twist = Twist()
      twist.angular.z = movement_vel
      #self.vel_publisher.publish(twist)
      self.publish_rate.sleep()

      #twist.angular.z = 0
      #self.vel_publisher.publish(twist)
      #self.publish_rate.sleep()
    else:
      rospy.loginfo("MOVING!!!!")


      movement_vel = - p_value_angular * angle_error
      '''
      if angle_error > 0.2:
        movement_vel =  - p_value_angular * angle_error
      elif angle_error < - 0.2:
        movement_vel =  - p_value_angular * angle_error
      else:
        movement_vel = 0
      '''
      twist = Twist()
      twist.angular.z = movement_vel

      #self.vel_publisher.publish(twist)
      #self.publish_rate.sleep()

      #twist = Twist()

      dist_error = length_offset - self.optimal_dist


      twist.linear.x = p_value_linear * dist_error

      self.vel_publisher.publish(twist)
      self.publish_rate.sleep()

      #twist = Twist()
      #twist.linear.x = 0

      #self.vel_publisher.publish(twist)
      #self.publish_rate.sleep()





if __name__=="__main__":
  DriveWheels()

