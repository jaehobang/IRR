#!/usr/bin/env python
# license removed for brevity
import os
import rospy
import cv2
import sys
import numpy as np
import math
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Point
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry



class goToGoal:


  def __init__(self):
    self.odom_subscriber = rospy.Subscriber("/odom", Odometry, self.update_odometry, queue_size = 1)
    self.way_points = []
    self.parse_txt()
    self.waypoint_counter = 0 #We will iterate through the way_points list with this counter






  def parse_txt(self):
    current_path = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(current_path, "wayPoints.txt")
    text_file = open(input_file, "r")
    lines = text_file.readlines()
    for line in lines:
      line = line.strip() #no more \n
      sentence_list = line.split(" ")
      if sentence_list[0] == "#" or sentence_list[0] == "":
        continue
      assert len(sentence_list) == 2
      self.way_points.append( (float(sentence_list[0]), float(sentence_list[1])) )


  def update_odometry(self, Odom):
    position = Odom.pose.pose.position

    # Orientation uses the quaternion aprametrization.
    # To get the angular position along the z-axis, the following equation is required.
    q = Odom.pose.pose.orientation
    orientation = np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))

    #frame_id of /odom = odom
    #child_frame_id of /odom = base_footprint

    if self.Init:
      # The initial data is stored to by subtracted to all the other values as we want to start at position (0,0) and orientation 0
      self.Init = False
      self.Init_ang = orientation
      self.globalAng = self.Init_ang
      Mrot = np.matrix(
        [[np.cos(self.Init_ang), np.sin(self.Init_ang)], [-np.sin(self.Init_ang), np.cos(self.Init_ang)]])
      self.Init_pos.x = Mrot.item((0, 0)) * position.x + Mrot.item((0, 1)) * position.y
      self.Init_pos.y = Mrot.item((1, 0)) * position.x + Mrot.item((1, 1)) * position.y
      self.Init_pos.z = position.z

    Mrot = np.matrix(
      [[np.cos(self.Init_ang), np.sin(self.Init_ang)], [-np.sin(self.Init_ang), np.cos(self.Init_ang)]])

    # We subtract the initial values
    self.globalPos.x = Mrot.item((0, 0)) * position.x + Mrot.item((0, 1)) * position.y - self.Init_pos.x
    self.globalPos.y = Mrot.item((1, 0)) * position.x + Mrot.item((1, 1)) * position.y - self.Init_pos.y
    self.globalAng = orientation - self.Init_ang

