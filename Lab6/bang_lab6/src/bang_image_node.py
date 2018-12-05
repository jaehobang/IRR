#!/usr/bin/env python

import cv2
import sys
import argparse
import csv
import numpy as np
import os
import rospy

from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String


class ImageNode:

    def __init__(self):
        """Train and wait for images to be passed
           Make the cropping and classification one pass
        """
        rospy.init_node('image_node', anonymous=True, log_level=rospy.DEBUG)
        rospy.loginfo("inside init...")
        self.train_data = None
        self.train_labels = None
        self.img = None
        self.x = 50
        self.y = 50
        self.knn = None
        self.img = None

        self.blur_size = 3
        self.lower = np.array([0, 80, 0], np.uint8)
        self.upper = np.array([180, 255, 255], np.uint8)
        self.val = np.array([0, 80, 0], np.uint8)
        self.error = np.array([13, 50, 50], np.uint8)
        self.morphOpSize = 5
        self.maxObjects = 1
        self.minObjectArea = 1500
        self.name = "Object!"

        # Color or grey classifier.
        self.colorBool = True

        # Number of neighbors to consider
        self.K = 3
        self.project_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        print(self.project_dir)
        self.train()


        # subscribes to images from camera
        #TODO: For simulation we have to have /raspicam_node/image/compressed/compressed
        #TODO: But for real one it should be /raspicam_node/image/compressed
        self.image_subscriber = rospy.Subscriber('/raspicam_node/image/compressed', CompressedImage, self.image_callback)
        # subscribes to requests from the main loop
        self.request_subscriber = rospy.Subscriber('/turtlebot3/request_classification', String, self.request_callback)

        # publishes the result of image classification
        self.classification_publisher = rospy.Publisher("/turtlebot3/classification", String, queue_size = 1)
        # publishes the result that we have finished training and we are ready to go
        self.image_setup_publisher = rospy.Publisher("/turtlebot3/image_node_setup", String, queue_size = 1)

	self.publish_rate = rospy.Rate(10)
        self.image_setup_publisher.publish( String("foo") )
	self.publish_rate.sleep()
        rospy.loginfo("Done init... now spinning...")
        rospy.spin()



    def _mouseEvent(self, event, x, y, flags, param):
        # The mouse event is connected to the "Original Image Window" and triggers the event when the user click on the image.
        # This event calculates the lower and upper bounds to filter the colors of the image

        if event == cv2.EVENT_LBUTTONDOWN:
            # lower[0] = img_hsv[y,x,0]
            # upper[0] = img_hsv[y,x,0]
            self.val = self.img_hsv[y, x, :]
            # upper = img_hsv[y,x,:]

            # lower = cv2.subtract(lower,error)
            # upper = cv2.add(upper,error)
            print("Hue color:")
            print("- pixel val:")
            print(self.val)
            # print("- Upper bound: ")
            # print(upper)

    def morphOps(self, binaryMatrix, kernelSize):
        # Morphological operations (open and close) used to reduce noise in the acquired image.
        kernel = np.ones((kernelSize, kernelSize), np.uint8)
        tempFix = cv2.morphologyEx(binaryMatrix, cv2.MORPH_CLOSE, kernel)  # Fill in holes
        fix = tempFix  # cv2.morphologyEx(tempFix,cv2.MORPH_OPEN, kernel)             # Get rid of noise
        return fix

    def drawCOM(self, frame, x, y, name):
        cv2.circle(frame, (x, y), 5, (0, 255, 0))
        cv2.putText(frame, name, (x - 30, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    def findObjects(self, binaryMatrix):
        # Finds the location of the desired object in the image.
        output = []
        trash, contours, hierarchy = cv2.findContours(binaryMatrix, cv2.RETR_EXTERNAL,
                                                      cv2.CHAIN_APPROX_SIMPLE)  # Contours the image to find blobs of the same color
        cont = sorted(contours, key=cv2.contourArea, reverse=True)[
               :self.maxObjects]  # Sorts the blobs by size (Largest to smallest)



        # Find the center of mass of the blob if there are any
        if hierarchy is not None:
            for i in range(0, len(cont)):
                M = cv2.moments(cont[i])
                if M['m00'] > self.minObjectArea:  # Check if the total area of the contour is large enough to care about!
                    x, y, w, h = cv2.boundingRect(cont[i])
                    rect = cv2.minAreaRect(cont[0])
                    w = int(rect[1][0])
                    x = int(M['m10'] / M['m00'])
                    y = int(M['m01'] / M['m00'])
                    # cv2.drawContours(imgTrack, cont[i], -1, (255,0,0), 3) # Draws the contour.
                    self.img_Crop = self.img_hsv[y - h / 2:y + h / 2, x - w / 2:x + w / 2].copy()
                    cv2.rectangle(self.imgTrack, (x - w / 2, y - h / 2), (x + w / 2, y + h / 2), (0, 255, 0), 2)
                    self.drawCOM(self.imgTrack, x, y, self.name)
                    if output == []:
                        output = [[x, w]]
                    else:
                        output.append[[x, w]]

        if len(output) is 0:  # We didn't find anything to track take whole image in (blank wall most likely)
            self.img_Crop = self.imgTrack.copy()

        return output

    def train(self):
        ### Load training images and labels
        with open( os.path.join(self.project_dir, 'imgs', 'train.txt'), 'rb') as train:
            reader_train = csv.reader(train)
            lines_train = list(reader_train)

        # read in training labels
        train_labels = np.array([np.int32(lines_train[i][1]) for i in range(len(lines_train))])


        train_list = []
        for i in range(0, len(lines_train)):
            img_bgr = cv2.imread(os.path.join(self.project_dir, 'imgs', lines_train[i][0] + ".png"), 1)
            self.imgTrack = img_bgr.copy()
            img_blur = cv2.GaussianBlur(img_bgr, (self.blur_size, self.blur_size), 0)
            self.img_hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(self.img_hsv, self.lower, self.upper)
            self.res = cv2.bitwise_and(self.imgTrack, self.imgTrack, mask=mask)
            imgMorphOps = self.morphOps(mask, self.morphOpSize)
            centers = self.findObjects(imgMorphOps)
            img_lil = cv2.resize(self.img_Crop, (40, 40))
            img_gray = cv2.cvtColor(img_lil, cv2.COLOR_BGR2GRAY)

            if (self.colorBool):
                train_list.append(np.array(img_lil))
            else:
                train_list.append(np.array(img_gray))


        train2 = np.asarray(train_list)
        if (self.colorBool):
            train_data = train2.flatten().reshape(len(lines_train), 40 * 40 * 3)
        else:
            train_data = train2.flatten().reshape(len(lines_train), 40 * 40)
        train_data = train_data.astype(np.float32)

        ### Train classifier
        self.knn = cv2.ml.KNearest_create()
        self.knn.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)

    ### Run test images
        with open(os.path.join(self.project_dir, 'imgs', 'test.txt'), 'rb') as f:
            reader = csv.reader(f)
            lines = list(reader)

        correct = 0.0
        confusion_matrix = np.zeros((6, 6))
        error_count = 0
        for i in range(len(lines)):
            #print("current line number is " + str(i))
            #print(str(lines[i]))
            try:
                img_bgr = cv2.imread( os.path.join(self.project_dir, 'imgs', lines[i][0] + ".png"), 1)
                if img_bgr is None: print("NOT ABLE TO READ IMAGE!!!!")
                self.imgTrack = img_bgr.copy()
                img_blur = cv2.GaussianBlur(img_bgr, (self.blur_size, self.blur_size), 0)
                self.img_hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(self.img_hsv, self.lower, self.upper)
                self.res = cv2.bitwise_and(self.imgTrack, self.imgTrack, mask=mask)
                imgMorphOps = self.morphOps(mask, self.morphOpSize)
                centers = self.findObjects(imgMorphOps)
                if self.img_Crop is None: print("Cropped image does not exist!!!")
                img_lil = cv2.resize(self.img_Crop, (40, 40))
                img_gray = cv2.cvtColor(img_lil, cv2.COLOR_BGR2GRAY)

                if (self.colorBool):
                    test_img = np.asarray(img_lil)
                    test_img = test_img.flatten().reshape(1, 40 * 40 * 3)
                else:
                    test_img = np.asarray(img_gray)
                    test_img = test_img.flatten().reshape(1, 40 * 40)
                test_img = test_img.astype(np.float32)
                test_label = np.int32(lines[i][1])

                ret, results, neighbours, dist = self.knn.findNearest(test_img, self.K)

                if test_label == ret:
                    print(lines[i][0], " Correct, ", ret)
                    correct += 1
                    confusion_matrix[np.int32(ret)][np.int32(ret)] += 1
                else:
                    confusion_matrix[test_label][np.int32(ret)] += 1

                    print(lines[i][0], " Wrong, ", test_label, " classified as ", ret)
                    print "\tneighbours: ", neighbours
            except:
		print("Error trying to open????")
                correct += 1
                error_count += 1

        print("Total accuracy: ", correct / len(lines))
        print(confusion_matrix)
        print("Error count: " + str(error_count))
        cv2.destroyAllWindows()


    def image_callback(self, data):
        self.img = data.data


    def request_callback(self, data):
        rospy.loginfo("request received!!")
        np_arr = np.fromstring(self.img, np.uint8)
        curr_image = cv2.imdecode(np_arr, 1)

        try:
            img_bgr = curr_image
            print("1")
            if img_bgr is None: print("NOT ABLE TO READ IMAGE!!!!")
            self.imgTrack = img_bgr.copy()
            print("1")

            img_blur = cv2.GaussianBlur(img_bgr, (self.blur_size, self.blur_size), 0)
            print("1")
            self.img_hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(self.img_hsv, self.lower, self.upper)
            self.res = cv2.bitwise_and(self.imgTrack, self.imgTrack, mask=mask)
            print("1")

            imgMorphOps = self.morphOps(mask, self.morphOpSize)
            print("1")

            centers = self.findObjects(imgMorphOps)
            if self.img_Crop is None: print("Cropped image does not exist!!!")
            img_lil = cv2.resize(self.img_Crop, (40, 40))
            img_gray = cv2.cvtColor(img_lil, cv2.COLOR_BGR2GRAY)
	    print("done with preprocessing")
            if (self.colorBool):
                test_img = np.asarray(img_lil)
                test_img = test_img.flatten().reshape(1, 40 * 40 * 3)
            else:
                test_img = np.asarray(img_gray)
                test_img = test_img.flatten().reshape(1, 40 * 40)
            test_img = test_img.astype(np.float32)
	    print("finished making test_img")

            ret, results, neighbours, dist = self.knn.findNearest(test_img, self.K)
	    print("returned from findNearest")


            self.classification_publisher.publish(str(ret))
            self.publish_rate.sleep()
        except:
	    self.classification_publisher.publish(str(-1))
	    self.publish_rate.sleep()
            print("Exception occurred while processing the image.....")




if __name__ == "__main__":
    imagenode = ImageNode()

