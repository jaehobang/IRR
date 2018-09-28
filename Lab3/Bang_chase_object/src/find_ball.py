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

    self.debug_mode = debug_mode
    if debug_mode:
      self.window_name = "image"
      cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
      cv2.startWindowThread()

    rospy.spin()

    if debug_mode:
      cv2.destroyAllWindows()


  def findBall(self, inputImage):

    height, width = inputImage.shape[:2]

    outputImage = cv2.resize(inputImage, (int(320.0 / height * width), 320))
    new_height, new_width = outputImage.shape[:2]
    imageBlur = cv2.GaussianBlur(outputImage, (11, 11), 3)
    imageHSV = cv2.cvtColor(imageBlur, cv2.COLOR_BGR2HSV)
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
    circles = cv2.HoughCircles(imageGray, cv2.HOUGH_GRADIENT, 1, 200, param1=70, param2=30, minRadius=30, maxRadius=500)

    if circles is not None:
      circles = np.round(circles[0, :]).astype("int")
      for (x, y, r) in circles:
        cv2.circle(outputImage, (x, y), r, (0, 255, 0), 4)

      return {"location": (x, y, new_width), "output": outputImage, "grayscale": imageGray,
              "segmentation": imageSegBRG, "blurred": imageBlur}

    else:
      return {"location": None, "output": None, "grayscale": imageGray,
              "segmentation": imageSegBRG, "blurred": imageBlur}


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
      #TODO: Need to make sure x = 0, y = 1
      location.x = result['location'][0]
      location.y = result['location'][1]
      location.z = result['location'][2]
      self.loc_publisher.publish(location)
    self.publish_rate.sleep()




  def displayImageStream(self, packedReturn, compressedImage):

    if packedReturn['output'] is not None:
      ballLocation = packedReturn['location']
      newImage = packedReturn['output']
      grayImage = packedReturn['grayscale']
      segImage = packedReturn['segmentation']
      blurImage = packedReturn['blurred']
      gray_3_channel = cv2.cvtColor(grayImage, cv2.COLOR_GRAY2BGR)
      cv2.imshow(self.window_name, np.vstack((blurImage, newImage, gray_3_channel)))
      print(ballLocation)
      # cv2.imshow(window_name, segImage)

    else:
      grayImage = packedReturn['grayscale']
      segImage = packedReturn['segmentation']
      blurImage = packedReturn['blurred']
      gray_3_channel = cv2.cvtColor(grayImage, cv2.COLOR_GRAY2BGR)
      height, width = compressedImage.shape[:2]

      resizedImage = cv2.resize(compressedImage, (int(320.0 / height * width), 320))
      cv2.imshow(self.window_name, np.vstack((blurImage, resizedImage, gray_3_channel)))

      print("N/A")
      # cv2.imshow(window_name, segImage)

    cv2.waitKey(0)



if __name__ == '__main__':
  args = sys.argv
  if len(args)  == 2 and args[1] == 'debug':
    BallFinder(True)
  else:
    BallFinder(False)


