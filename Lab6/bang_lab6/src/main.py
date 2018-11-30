#!/usr/bin/env python
# license removed for brevity
import os
import rospy
import tf
import signal
import math
import numpy as np

from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from move_base_msgs.msg import MoveBaseActionResult
from geometry_msgs.msg import Point
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import CompressedImage

from geometry_msgs.msg import Twist
from std_msgs.msg import String



"""
Basically, If you see arrow, move 90 degrees in that direction
After moving 90 degrees, if there is a wall in front of you, you should read the sign
If there is not a wall in front of you, you should move forward until you are 1 foot away from a wall, then stop
When you are stopped, you should use the camera to read the sign

If you see stop sign, turn 180 degrees and move forward until you are 1 foot against a wall
If you see ! sign, then exit
If you are too close to the wall, move backwards until you are 1 foot away from the wall

Topics to listen to:
compressed_images
scan


Topics to publish to:
cmd_vel

Services to rely on:
send an compressed image,
receive a number - it tells what sign it is
"""



class goToGoal:

  def __init__(self):
    rospy.init_node('main', anonymous=True, log_level=rospy.DEBUG, disable_signals = True)
    self.scan_subscriber = rospy.Subscriber("/scan", LaserScan, self.callback_laser, queue_size=1)
    self.classification_subscriber = rospy.Subscriber("/turtlebot3/classification", String, self.classification_callback, queue_size=1)

    self.vel_publisher = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
    self.classification_publisher = rospy.Publisher("/turtlebot3/request_classification", String, queue_size = 1)


    self.publish_rate = rospy.Rate(10)
    self.turn_rate = rospy.Rate(1)

    self.check_direction = None



    self.init_scheme()

    rospy.on_shutdown(self.stop_wheels)
    signal.signal(signal.SIGINT, self.signal_handler)
    rospy.spin()

  def signal_handler(self, sig, frame):
    rospy.loginfo('You pressed Ctrl+C!')
    self.stop_wheels()
    rospy.signal_shutdown("Shutdown initiated..")

  def callback_laser(self, data):
    #This is when we have arrived at a stop after turning and need to check for delta turning
    if self.check_direction is not None:
      if self.check_direction == "left":
        length = len(data.ranges)
        right_limit = 8 * length / 36
        left_limit = 10 * length / 36

      if self.check_direction == "right":
        length = len(data.ranges)
        right_limit = 26 * length / 36
        left_limit = 28 * length / 36

      left_range = np.array(data.ranges[left_limit:left_limit + 5])
      right_range = np.array(data.ranges[right_limit - 5:right_limit])
      left_range[left_range < data.range_min] = 0
      right_range[right_range < data.range_min] = 0
      left_avg = sum(left_range) / len(left_range[left_range != 0])
      right_avg = sum(right_range) / len(right_range[right_range != 0])

      eval_return = self.evaluate_with_epsilon(left_avg, right_avg)
      if eval_return == "left":
        self.turn_a_bit("right") #TODO
      elif eval_return == "right":
        self.turn_a_bit("left") #TODO
      else: #eval_return == "equal"
        assert(eval_return == "equal")
        self.stop_wheels()
        self.check_direction = None
    #this is when we are going forward
    else:
      length = len(data.ranges)
      left_limit = length / 36  # we will look at 0 - 10 degrees and 350 - 360 degrees
      points_of_interest = np.array(data.ranges[:left_limit][::-1] + data.ranges[-left_limit:][::-1])

      points_of_interest[points_of_interest < data.range_min] = 0
      wall_dist = sum(points_of_interest) / len(points_of_interest[points_of_interest != 0])

      if wall_dist > 0.3: # one foot away from the sign
        self.keep_going() #TODO
      else:
        self.stop_wheels()
        self.request_classification() #TODO

  def turn_a_bit(self, direction):
    twist = Twist()
    if direction == "right":
      twist.z = -0.5
    elif direction == "left":
      twist.z = 0.5
    self.vel_publisher.publish(twist)
    self.publish_rate.sleep()

  def evaluate_with_epsilon(self, left_v, right_v):
    epsilon = 0.05
    if abs(left_v - right_v) < epsilon:
      return "equal"
    elif left_v - right_v > epsilon:
      return "left"
    else:
      return "right"

  def keep_going(self):
    twist = Twist()
    twist.x = 0.5

    self.vel_publisher.publish(twist)
    self.publish_rate.sleep()

  def stop_wheels(self):
    twist = Twist()
    twist.x = 0
    self.vel_publisher.publish(twist)
    self.publish_rate.sleep()

  def turn_90_degrees(self, direction):
    twist = Twist()
    if direction == 'left':

      twist.z = 1.57
    if direction == 'right':
      twist.z = -1.57

    self.vel_publisher.publish(twist)
    self.turn_rate.sleep()

    #Need to confirm that it actually did turn 90 degrees, we can check it with scan messages
    if direction == 'left':
      self.check_direction = 'right'
    if direction == 'right':
      self.check_direction = 'left'

  def turn_180_degrees(self):
    twist = Twist()
    twist.z = 1.57
    self.vel_publisher.publish(twist)
    for i in xrange(2): self.turn_rate.sleep()
    #Is there a way to check turn rate delta??? skip for now

  def request_classification(self):

    str = String("foo")
    self.classification_publisher.publish(str)
    self.publish_rate.sleep()

  def get_classification_and_drive(self, data):
    # data should be a number according to what is available in the train.txt, test.txt
    #   -> we will convert it in this function
    # '0' -> no sign
    # '1' -> move_left
    # '2' -> move_right
    # '3' -> stop (do_not_enter but does the same thing)
    # '4' -> stop
    # '5' -> goal (exit_program)
  # "move_left", "move_right", "stop", "done"
    if data.data == '0':
      self.request_classification()
    elif data.data == '1':
      self.turn_90_degrees('left')
    elif data.data == '2':
      self.turn_90_degrees('right')
    elif data.data == '3' or data.data == '4':
      self.turn_180_degrees()
    elif data.data == '5':
      self.stop_wheels()
      rospy.signal_shutdown("Reached Goal!")

  def init_scheme(self): #Here we will wait for image node to finish its classification process
    rospy.loginfo("Waiting for message from image classifier...")
    rospy.wait_for_message("/turtlebot3/image_node_setup", String)
    rospy.loginfo("Message received, training done, we are ready to go")






if __name__ == "__main__":
  goToGoal()
