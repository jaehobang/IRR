#!/usr/bin/env python
# license removed for brevity
import rospy
import signal
import numpy as np
import random

from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from move_base_msgs.msg import MoveBaseActionResult
from sensor_msgs.msg import LaserScan

from geometry_msgs.msg import Twist
from std_msgs.msg import String

from tf.transformations import quaternion_from_euler, euler_from_quaternion



"""
We will map the arena
We will use move_base
we will use navigation stack

we need occupany grid size
we need to stay in middle at all times when turning

1. Localize
while not done
    1. detect dist to front wall
    2. calculate dist need to move in x direction for image detection
    3. trigger move_base to move to point
    4. detect image
    5. turn accordingly (might need to move back)

"""



class goToGoal:

    def __init__(self):
        rospy.init_node('main', anonymous=True, log_level=rospy.DEBUG, disable_signals = True)
        # pubs and subs
        #self.initial_publisher = rospy.Publisher("/initialpose", PoseWithCovarianceStamped, queue_size=1)
        self.goal_publisher = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=1)
        self.classification_publisher = rospy.Publisher("/turtlebot3/request_classification", String, queue_size = 1)
        self.vel_publisher = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

        self.result_subscriber = rospy.Subscriber("/move_base/result", MoveBaseActionResult, self.update_goal, queue_size=1)
        self.scan_subscriber = rospy.Subscriber("/scan", LaserScan, self.callback_laser, queue_size=1)
        self.classification_subscriber = rospy.Subscriber("/turtlebot3/classification", String, self.classification_callback, queue_size=1)

        self.publish_rate = rospy.Rate(10)

        # variables
        self.grid_size = 1.0
        self.grid_center = self.grid_size / 2
        self.dist_to_image = 0.35
        self.dist_to_wall_front = -1
        self.dist_to_wall_left = -1
        self.dist_to_wall_right = -1

        self.current_pose = None

        self.global_seq = 0
        self.wait = True #waiting for a reply from action
        self.command = None #classification result from imagenode
        self.direction = [0, 1.57, 3.14, 4.71]
        self.d_name = ['up', 'left', 'down', 'right']

        rospy.on_shutdown(self.stop_wheels)
        signal.signal(signal.SIGINT, self.signal_handler)

        self.run_loop()

        rospy.spin()

    def stop_wheels(self):
        twist = Twist()
        self.vel_publisher.publish(twist)
        self.publish_rate.sleep()


    def run_loop(self):
        self.current_pose = rospy.wait_for_message("/initialpose", PoseWithCovarianceStamped)
        goal_reached = False
        while not goal_reached:
            self.move_to_wall_front()
            self.wait_for_action_finish()
            self.request_classification()
            classification_result = self.wait_for_classification_finish()
            if classification_result == "no_sign":
                self.wiggle()
            elif classification_result == "move_left":
                self.turn("left")
            elif classification_result == "move_right":
                self.turn("right")
            elif classification_result == "stop":
                self.turn("flip")
            elif classification_result == "goal":
                rospy.signal_shutdown("We have reached our destination!")
            self.wait_for_action_finish()

    def turn(self, dir):
	rospy.loginfo("Inside function turn")
        x = self.current_pose.pose.pose.position.x
        y = self.current_pose.pose.pose.position.y
        _,_,theta = euler_from_quaternion([self.current_pose.pose.pose.orientation.x,
                                          self.current_pose.pose.pose.orientation.y,
                                          self.current_pose.pose.pose.orientation.z,
                                          self.current_pose.pose.pose.orientation.w])
        curr_heading = self._determine_direction(theta)
	theta_copy = theta #only used for debugging purposes
        if dir == "left":
            theta += 1.57
        elif dir == "right":
            theta -= 1.57
        elif dir == "flip":
            theta += 3.14

	

        if curr_heading == "left":
            y -= self.grid_center - self.dist_to_image
        elif curr_heading == "up":
            x -= self.grid_center - self.dist_to_image
        elif curr_heading == "down":
            x += self.grid_center - self.dist_to_image
        elif curr_heading == "right":
            y += self.grid_center - self.dist_to_image

	rospy.loginfo("offset to dist movement for turning is" + 
			str(self.grid_center - self.dist_to_image))

        new_objective_ = PoseStamped()
        new_objective_.header.seq = self.global_seq
        new_objective_.header.stamp = rospy.Time.now()
        new_objective_.header.frame_id = "map"  # not sure if this matters


        tx, ty, tz, tw = quaternion_from_euler(0, 0, theta)
        new_objective_.pose.position.x = x
        new_objective_.pose.position.y = y
        new_objective_.pose.orientation.x = tx
        new_objective_.pose.orientation.y = ty
        new_objective_.pose.orientation.z = tz
        new_objective_.pose.orientation.w = tw

	rospy.loginfo("Robot is currently facing " + curr_heading)
	rospy.loginfo("Current position is %f %f %f" % (self.current_pose.pose.pose.position.x,
							self.current_pose.pose.pose.position.y,
							theta_copy))
	rospy.loginfo("Objtive position is %f %f %f" % (new_objective_.pose.position.x,
							new_objective_.pose.position.y,
							theta))


	self.current_pose.pose.pose.position.x = x
	self.current_pose.pose.pose.position.y = y
	self.current_pose.pose.pose.orientation.x = tx
	self.current_pose.pose.pose.orientation.y = ty
	self.current_pose.pose.pose.orientation.z = tz
	self.current_pose.pose.pose.orientation.w = tw

        self.goal_publisher.publish(new_objective_)
        self.publish_rate.sleep()
        self.global_seq += 1


    def get_current_pose(self):
        """
        TODO: Need to find the topic that tells you the current position and set self.current_pose to it
        :return: None
        """
        self.current_pose = self.current_pose

    def wiggle(self):
        """
        Hopefully, we will never have to run this function
        Currently, just stay in place
        :return:
        """
        rospy.loginfo("UGHHHHHHH image node can't detect sign")
        new_objective_ = PoseStamped()
        new_objective_.header.seq = self.global_seq
        new_objective_.header.stamp = rospy.Time.now()
        new_objective_.header.frame_id = "map"  # not sure if this matters
        new_objective_.pose.position.x = self.current_pose.pose.pose.position.x
        new_objective_.pose.position.y = self.current_pose.pose.pose.position.y

	d = [(0, 0, 0, 1), (0, 0, 0.707, 0.707), (0, 0, 1, 0), (0, 0, -0.707, 0.707)]
	d_name = ['up', 'left', 'down', 'right']
	rand_num = random.randint(0,4)
	new_ori = d[rand_num]
	rospy.loginfo("Inside wiggle trying " + d_name[rand_num])
	

        new_objective_.pose.orientation.x = new_ori[0]
        new_objective_.pose.orientation.y = new_ori[1]
        new_objective_.pose.orientation.z = new_ori[2]
        new_objective_.pose.orientation.w = new_ori[3]


	self.current_pose.pose.pose.orientation.x = new_ori[0]
	self.current_pose.pose.pose.orientation.y = new_ori[1]
	self.current_pose.pose.pose.orientation.z = new_ori[2]
	self.current_pose.pose.pose.orientation.w = new_ori[3]

        self.global_seq += 1
        self.goal_publisher.publish(new_objective_)
        self.publish_rate.sleep()

    def wait_for_classification_finish(self):
        while self.command == None: pass
        classification_result = self.command
        self.command = None
        return classification_result

    def classification_callback(self, data):
        # data should be a number according to what is available in the train.txt, test.txt
        #   -> we will convert it in this function
        # '0' -> no_sign
        # '1' -> move_left
        # '2' -> move_right
        # '3' -> stop (do_not_enter but does the same thing)
        # '4' -> stop
        # '5' -> goal (exit_program)
        # "move_left", "move_right", "stop", "done"
        rospy.loginfo("classification result is" + data.data)

        num = eval(data.data)
        labels = ["no_sign", "move_left", "move_right", "stop", "stop", "goal"]
        self.command = labels[int(num)]


    def request_classification(self):
        self.classification_publisher.publish(String("foo"))
        self.publish_rate.sleep()


    def update_goal(self, data):
	SUCCEEDED = 3
        rospy.loginfo(str(data.status.status))
        if data.status.status == SUCCEEDED:
            self.wait = False

    def wait_for_action_finish(self):
        while(self.wait): pass
        self.wait = True

    def move_to_wall_front(self):
        """
        This function should be only called when you are trying to move forward
        """
        x = self.current_pose.pose.pose.position.x
        y = self.current_pose.pose.pose.position.y
        _,_, theta = euler_from_quaternion([self.current_pose.pose.pose.orientation.x,
                                           self.current_pose.pose.pose.orientation.y,
                                           self.current_pose.pose.pose.orientation.z,
                                           self.current_pose.pose.pose.orientation.w])
        dir = self._determine_direction(theta)
	rospy.loginfo("Inside move to wall front , current heading is " + dir)
        if dir == 'up':
            new_objective = (x + self.dist_to_wall_front - self.dist_to_image,
                             y)
        elif dir == 'left':
            new_objective = (x,
                             y + self.dist_to_wall_front - self.dist_to_image)
        elif dir == 'right':
            new_objective = (x,
                             y - (self.dist_to_wall_front - self.dist_to_image))
        elif dir == 'down':
            new_objective = (x - (self.dist_to_wall_front - self.dist_to_image),
                             y)
        else:
            rospy.loginfo("self._determine_direction giving wrong direction...")

        tx,ty,tz,tw = quaternion_from_euler(0, 0, theta)

        new_objective_ = PoseStamped()
        new_objective_.header.seq = self.global_seq
        new_objective_.header.stamp = rospy.Time.now()
        new_objective_.header.frame_id = "map" #not sure if this matters
        new_objective_.pose.position.x = new_objective[0]
        new_objective_.pose.position.y = new_objective[1]
        new_objective_.pose.orientation.x = tx
        new_objective_.pose.orientation.y = ty
        new_objective_.pose.orientation.z = tz
        new_objective_.pose.orientation.w = tw

	rospy.loginfo("Robot is currently facing " + dir)
	rospy.loginfo("Current position is %f %f %f" % (self.current_pose.pose.pose.position.x,
							self.current_pose.pose.pose.position.y,
							theta))
	rospy.loginfo("Objtive position is %f %f %f" % (new_objective_.pose.position.x,
							new_objective_.pose.position.y,
							theta))


        self.global_seq += 1
        self.current_pose.pose.pose.position.x = new_objective_.pose.position.x
        self.current_pose.pose.pose.position.y = new_objective_.pose.position.y
        self.current_pose.pose.pose.orientation.x = tx
        self.current_pose.pose.pose.orientation.y = ty
        self.current_pose.pose.pose.orientation.z = tz
        self.current_pose.pose.pose.orientation.w = tw

        self.goal_publisher.publish(new_objective_)
        self.publish_rate.sleep()


    def _determine_direction(self, theta):
        """
        TODO: Need to determine the directions are actually correct from experiment
        :param theta: in radians (yaw)
        :return: 'up', 'left', 'right', 'down' where up refers to x+ and left refers to y+
        """

	theta = theta % 6.28
        error = 0.15 #+- 30 degrees
        for i in xrange(len(self.direction)):
            if theta > self.direction[i] - error and theta < self.direction[i] + error:
                return self.d_name[i]

	rospy.loginfo("Inside _determine_direction we should never reach here...")
	rospy.loginfo("theta value is " + str(theta))



    def signal_handler(self, sig, frame):
        rospy.loginfo('You pressed Ctrl+C!')
        self.stop_wheels()
        rospy.signal_shutdown("Shutdown initiated..")


    def callback_laser(self, data):
        length = len(data.ranges)

        left_limit = length / 36
        points_of_interest = np.array(data.ranges[:left_limit][::-1] + data.ranges[-left_limit:][::-1])
        points_of_interest[points_of_interest < data.range_min] = 0
        points_of_interest[points_of_interest > data.range_max] = 0
        denom = len(points_of_interest[points_of_interest != 0])
        if denom == 0: denom = 1
        self.dist_to_wall_front = sum(points_of_interest) / denom


        right_limit = 8 * length / 36
        left_limit = 10 * length / 36
        points_of_interest = np.array(data.ranges[right_limit:left_limit][::-1])
        points_of_interest[points_of_interest < data.range_min] = 0
        points_of_interest[points_of_interest > data.range_max] = 0
        denom = len(points_of_interest[points_of_interest != 0])
        if denom == 0: denom = 1
        self.dist_to_wall_left = sum(points_of_interest) / denom


        right_limit = 26 * length / 36
        left_limit = 28 * length / 36
        points_of_interest = np.array(data.ranges[right_limit:left_limit][::-1])
        points_of_interest[points_of_interest < data.range_min] = 0
        points_of_interest[points_of_interest > data.range_max] = 0
        denom = len(points_of_interest[points_of_interest != 0])
        if denom == 0: denom = 1
        self.dist_to_wall_right = sum(points_of_interest) / denom









if __name__ == "__main__":
  goToGoal()
