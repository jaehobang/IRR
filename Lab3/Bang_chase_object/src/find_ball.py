#!/usr/bin/env python
# license removed for brevity
import rospy
import cv2
import sys
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Point


class BallFinder:


  def __init__(self, debug_mode):
    rospy.init_node('ballFinder', anonymous = True)
    rospy.Subscriber("/raspicam_node/image/compressed", CompressedImage, self.callback)
    self.loc_publisher = rospy.Publisher("/turtlebot3/burger/ball_location", Point, queue_size = 1)
    self.publish_rate = rospy.Rate(10)  # 10hz
    #self.publish_image_rate = rospy.Rate(1) #1hz

    self.debug_mode = debug_mode
    if debug_mode:
      self.seq = 0
      self.image_publisher = rospy.Publisher("/turtlebot3/burger/image_raw/compressed", CompressedImage, queue_size = 1)
    rospy.spin()

    if debug_mode:
      cv2.destroyAllWindows()


  def findBall(self, inputImage):

    height, width = inputImage.shape[:2]

    outputImage = cv2.resize(inputImage, (int(320.0 / height * width), 320))
    new_height, new_width = outputImage.shape[:2]
    #imageBlur = cv2.GaussianBlur(outputImage, (11, 11), 3)
    imageHSV = cv2.cvtColor(outputImage, cv2.COLOR_BGR2HSV)
    lowerPink = np.array([145, 10, 75])  # pink:[145, 10, 75], [180, 255, 255]
    upperPink = np.array([180, 255, 255]) # yellow:[20, 100, 100], [30, 255, 255]
    maskPink = cv2.inRange(imageHSV, lowerPink, upperPink)
    imageSeg = cv2.bitwise_and(imageHSV, imageHSV, mask=maskPink)
    imageSegBRG = cv2.cvtColor(imageSeg, cv2.COLOR_HSV2BGR)
    imageGray = cv2.cvtColor(imageSegBRG, cv2.COLOR_BGR2GRAY)

    """
    HoughCircles(image, method, dp, minDist) documentation

    image - input image in grayscale
    method - method of detecting circles, 
    dp - inverse ratio of the accumulator resolution
    minDist - distance between the circle centers
    param1 - gradient value used in paper
    param2 - threshold value -> smaller the threshold, the more circles will be detected
    minRadius - minimum size of radius in pixels
    maxRaius - maximum size of radius in pixels
    """
    circles = cv2.HoughCircles(imageGray, cv2.HOUGH_GRADIENT, 1, 200, param1=70, param2=25, minRadius=30, maxRadius=500)

    if circles is not None:
      circles = np.round(circles[0, :]).astype("int")
      for (x, y, r) in circles:
        cv2.circle(outputImage, (x, y), r, (0, 255, 0), 4)

      return {"location": (x, r, new_width), "output": outputImage}

    else:
      return {"location": None, "output": outputImage}


  def callback(self, data):
    # Retrieve the image, find the location, publish that
    #rospy.logerr(type(data.format))
    #rospy.logerr(data.format)
    #rospy.logerr(type(data.data))
    #rospy.logerr(data.data)
    #tmp = []
    #assert(type(compressedImage) == type(tmp))
    #assert(type(compressedImage) == type("hello world"))
    
    np_arr = np.fromstring(data.data, np.uint8)
    compressedImage = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Need to see if I can run this matrix into the algorithm I have developed for lab1
    result = self.findBall(compressedImage)

    if self.debug_mode:
      self.displayImageStream(result, compressedImage)

    if result['location'] is not None:
      location = Point()
      #BIG NOTE: when the robot is facing me... the robot's right is a low pixel value,
                                              # the robot's left is a high pixel value
      location.x = result['location'][0] #x
      location.y = result['location'][1] #r
      location.z = result['location'][2] #new_width
      self.loc_publisher.publish(location)
    self.publish_rate.sleep()




  def displayImageStream(self, packedReturn, compressedImage):
    outputImage = packedReturn['output']
    image = CompressedImage()
    image.header.seq = self.seq
    self.seq += 1
    image.header.stamp = rospy.Time.now()
    image.header.frame_id = "blab"
    image.format = "jpeg"

    image.data = np.array(cv2.imencode('.jpg', outputImage)[1]).tostring()

    self.image_publisher.publish(image)
    #self.publish_image_rate.sleep()



if __name__ == '__main__':
  args = sys.argv
  if len(args)  == 2 and args[1] == 'debug':
    BallFinder(True)
  else:
    BallFinder(False)


