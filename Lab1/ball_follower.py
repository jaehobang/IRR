#! usr/bin/env python


import cv2
import numpy as np

"""Directions:
   1. Successfully run the code on the robot - ROS related
   2. Find specific ball in image 50% or more of the time
   3. Print the location of the ball
   4. Display the location of the ball on the image - draw a box around it? """


# I want to test this on laptop before putting it on the robot.
# find_ball will have a wrapper that deals with ROS communication


def test_laptop_webcam():
  cap = cv2.VideoCapture(0)
  while (True):
    ret, frame = cap.read()

    testImage = frame
    packedReturn = find_ball(testImage)
    window_name = "image"
    cv2.namedWindow(window_name, cv2.WND_PROP_AUTOSIZE)
    cv2.startWindowThread()

    if packedReturn is not None:
      ballLocation, newImage = packedReturn
      cv2.imshow(window_name, newImage)
    else:
      cv2.imshow(window_name, testImage)

    if cv2.waitKey(5) & 0xFF == ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows()




# this script will simply find the location of the ball, and output the annotated image
# ball will be pink
def find_ball(inputImage):
  """This script should receive images from the webcam on your laptop
   and track the location of a specific ball in the frame.
   Once you have made the script it is often useful to make it an executable.
   To do this you must first make sure you have the type of environment 
   being used in the first line of your code"""

  outputImage = inputImage.copy()
  imageGray = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)

  circles = cv2.HoughCircles(imageGray, cv2.HOUGH_GRADIENT, 1, 90, param1=70, param2=60, minRadius=50, maxRadius=100)

  if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
      cv2.circle(outputImage, (x, y), r, (0, 255, 0), 4)

    return [(x,y), outputImage]

  return None



def info():
  """After grabbing the image, process the image to look for circles.
An easy place to start is to use theHoughCircles() functionality within OpenCV:


circles = cv2.HoughCircles(cv_image, cv2.HOUGH_GRADIENT, 1, 90, param1=70, param2=60,minRadius=50, maxRadius=0)

Normalizing and slightly blurring the image c


Once you’ve located the ball in an image, this script should print the pixel coordinate of the balland display some sort
of marker (or text pixel location) on the image itself.
"""
  pass


if __name__=="__main__":
  test_laptop_webcam()
