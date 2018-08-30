#!/usr/bin/env python
"""
Code written by Jaeho Bang for Intro to Robotics Research Class Fall 2018
"""
import cv2
import numpy as np


def test_laptop_webcam():
  cap = cv2.VideoCapture(0)
  while (True):
    ret, frame = cap.read()

    testImage = frame
    packedReturn = find_ball(testImage)
    window_name = "image"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.startWindowThread()

    if packedReturn['output'] is not None:
      ballLocation = packedReturn['location']
      newImage = packedReturn['output']
      grayImage = packedReturn['grayscale']
      segImage = packedReturn['segmentation']
      blurImage = packedReturn['blurred']
      gray_3_channel = cv2.cvtColor(grayImage, cv2.COLOR_GRAY2BGR)
      cv2.imshow(window_name, np.vstack((blurImage, newImage, gray_3_channel)))
      print(ballLocation)
      #cv2.imshow(window_name, segImage)

    else:
      grayImage = packedReturn['grayscale']
      segImage = packedReturn['segmentation']
      blurImage = packedReturn['blurred']
      gray_3_channel = cv2.cvtColor(grayImage, cv2.COLOR_GRAY2BGR)
      height, width = testImage.shape[:2]

      resizedImage = cv2.resize(testImage, (int(320.0 / height * width), 320))
      cv2.imshow(window_name, np.vstack((blurImage, resizedImage, gray_3_channel)))
      print("N/A")
      #cv2.imshow(window_name, segImage)

    if cv2.waitKey(5) & 0xFF == ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows()


def find_ball(inputImage):

  height, width = inputImage.shape[:2]

  outputImage = cv2.resize(inputImage, (int(320.0 / height * width), 320))
  imageBlur = cv2.GaussianBlur(outputImage, (11,11), 3)
  imageHSV = cv2.cvtColor(imageBlur, cv2.COLOR_BGR2HSV)
  lowerPink = np.array([145, 10, 75]) #[145, 75, 75]
  upperPink = np.array([180, 255, 255])
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

    return {"location": (x,y), "output": outputImage, "grayscale": imageGray,
            "segmentation": imageSegBRG, "blurred":imageBlur}

  else:
    return {"location": None, "output": None, "grayscale": imageGray,
            "segmentation": imageSegBRG, "blurred":imageBlur}



if __name__=="__main__":
  test_laptop_webcam()
