#!/usr/bin/env python
# license removed for brevity
import os
import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from move_base_msgs.msg import MoveBaseActionResult




class goToGoal:

  def __init__(self):
    rospy.init_node('main', anonymous=True, log_level=rospy.DEBUG, disable_signals = True)
    self.initial_publisher = rospy.Publisher("/initialpose", PoseWithCovarianceStamped, queue_size = 1)
    self.goal_publisher = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size = 1)
    self.result_subscriber = rospy.Subscriber("/move_base/result", MoveBaseActionResult, self.update_goal, queue_size = 1)
    self.publish_rate = rospy.Rate(10)
    self.rest_rate = rospy.Rate(0.5) # Wait for 2 seconds when you reach the goal


    self.init_scheme()



    rospy.spin()

  def init_pos(self):
    initial_pos = PoseWithCovarianceStamped()
    initial_pos.header.frame_id = "map"
    initial_pos.header.seq = 0
    initial_pos.header.stamp = rospy.Time.now()
    #TODO: Need to fix these values
    initial_pos.pose.pose.position.x = 0
    initial_pos.pose.pose.position.y = 0
    initial_pos.pose.pose.position.z = 0
    initial_pos.pose.pose.orientation.x = 0
    initial_pos.pose.pose.orientation.y = 0
    initial_pos.pose.pose.orientation.z = 0
    initial_pos.pose.pose.orientation.w = 0

    initial_pos.covariance = [0.25, 0, 0, 0, 0, 0,
                               0, 0.25, 0, 0, 0, 0,
                               0,    0, 0, 0, 0, 0,
                               0,    0, 0, 0, 0, 0,
                               0,    0, 0, 0, 0, 0,
                               0,    0, 0, 0, 0, 0.685]

    return initial_pos


  def send_goal(self):
    curr_waypoint = self.way_points[self.current_waypoint]
    goal = PoseStamped()
    goal.header.frame_id = "map"
    goal.header.seq = self.goal_seq
    self.goal_seq += 1
    goal.header.stamp = rospy.Time.now()
    goal.pose.position.x = curr_waypoint[0]
    goal.pose.position.y = curr_waypoint[1]
    goal.pose.position.z = curr_waypoint[2]
    goal.pose.orientation.x = curr_waypoint[3]
    goal.pose.orientation.y = curr_waypoint[4]
    goal.pose.orientation.z = curr_waypoint[5]
    goal.pose.orientation.w = curr_waypoint[6]

    self.goal_publisher.publish(goal)
    return

  def init_scheme(self):
    self.way_points = []
    self.current_waypoint = 0
    self.waypoint_counter = 0  # We will iterate through the way_points list with this counter
    self.goal_seq = 0
    self.parse_txt()

    #Wait for navigation stack to be fully up before publishing
    while self.initial_publisher.get_num_connections() == 0 or self.goal_publisher.get_num_connections() == 0:
        pass

    self.initial_publisher.publish( self.init_pos() )
    self.rest_rate.sleep()
    self.send_goal()
    self.publish_rate.sleep()


  def parse_txt(self):
    current_path = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(current_path, "global_waypoints.txt")
    text_file = open(input_file, "r")
    lines = text_file.readlines()
    for line in lines:
      line = line.strip() #no more \n
      sentence_list = line.split(" ")
      if sentence_list[0] == "#" or sentence_list[0] == "":
        continue
      assert len(sentence_list) == 7
      self.way_points.append( map(float, sentence_list) )

  def update_goal(self):
    if self.current_waypoint == len(self.way_points):
        exit()
    self.current_waypoint += 1
    #Need to rest for two seconds in between reaching the waypoints
    self.rest_rate.sleep()
    self.send_goal()





if __name__ == "__main__":
  goToGoal()
