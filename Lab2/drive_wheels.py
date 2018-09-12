import rospy
import cv2
import sys
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Point
from geometry_msgs.msg import Twist



class DriveWheels:
  def __init__(self, debug_mode):
    rospy.init_node('ballFinder', anonymous = True)
    rospy.Subscriber("/turtlebot3/burger/ball_location", Point, self.callback)
    self.vel_publisher = rospy.Publisher("turtlebot_telop_keyboard/cmd_vel", Twist, queue_size = 1)
    self.publish_rate = rospy.Rate(10)  # 10hz

    self.debug_mode = debug_mode

    self.center_points = []
    self.P = 0.1

    rospy.spin()


  def callback(self, data):
    self.center_points.append(data)

    previous_point = data[-2]
    current_point = data[-1]
    x_offset = (current_point[0] - previous_point[0]) #if positive, need to move to right
    #TODO: Need to debug this step and see if applying to yaw is the correct way to go
    movement_vel = self.P * x_offset
    twist = Twist()
    twist.angular.z = movement_vel

    self.vel_publisher.publish(twist)
    self.publish_rate.sleep()




if __name__=="__main__":
  DriveWheels()

