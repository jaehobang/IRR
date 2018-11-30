#!/usr/bin/env python

import cv2
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
        rospy.init_node('image_node')
        self.train_data = None
        self.train_labels = None
        self.img = None
        self.x = 50
        self.y = 50
        self.knn = None
        self.img = None
        if not os.path.isdir("./imgs_mod"):
          os.mkdir("imgs_mod")

        self.train()

        # subscribes to images from camera
        self.image_subscriber = rospy.Subscriber('/raspicam_node/image/compressed', CompressedImage, self.image_callback)
        # subscribes to requests from the main loop
        self.request_subscriber = rospy.Subscriber('/turtlebot3/request_classification', String, self.request_callback)

        # publishes the result of image classification
        self.classification_publisher = rospy.Publisher("/turtlebot3/classification", String, queue_size = 1)
        # publishes the result that we have finished training and we are ready to go
        self.image_setup_publisher = rospy.Publisher("/turtlebot3/image_node_setup", String, queue_size = 1)

        self.image_setup_publisher.publish( String("foo") )
        rospy.spin()


    def image_callback(self, data):
        self.img = data.data


    def sort_contours(self, cnts, method="left-to-right"):
        # initialize the reverse flag and sort index
        reverse = False
        i = 0

        # handle if we need to sort in reverse
        if method == "right-to-left" or method == "bottom-to-top":
            reverse = True

        # handle if we are sorting against the y-coordinate rather than
        # the x-coordinate of the bounding box
        if method == "top-to-bottom" or method == "bottom-to-top":
            i = 1

        # construct the list of bounding boxes and sort them from top to
        # bottom
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                            key=lambda b: b[1][i], reverse=reverse))

        # return the list of sorted contours and bounding boxes
        return (cnts, boundingBoxes)



    def preprocess_on_fly(self, img):

        height, width = img.shape

        (thresh, img_bin) = cv2.threshold(img, 100, 255,
                                          cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # Thresholding the image
        img_bin = 255 - img_bin  # Invert the image

        # Defining a kernel length
        kernel_length = np.array(img).shape[1] // 40

        # A verticle kernel of (1 X kernel_length), which will detect all the verticle lines from the image.
        verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
        # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
        hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
        # A kernel of (3 X 3) ones.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # Morphological operation to detect verticle lines from an image
        img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=3)
        verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3)

        # Morphological operation to detect horizontal lines from an image
        img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)
        horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)
        # Weighting parameters, this will decide the quantity of an image to be added to make a new image.
        alpha = 0.5
        beta = 1.0 - alpha
        # This function helps to add two image with specific weight parameter to get a third image as summation of two image.
        img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
        img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
        (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # For Debugging
        # Enable this line to see verticle and horizontal lines in the image which is used to find boxes
        # Find contours for image, which will detect all the boxes
        im2, contours, hierarchy = cv2.findContours(
          img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Sort all the contours by top to bottom.
        (contours, boundingBoxes) = self.sort_contours(contours, method="top-to-bottom")

        idx = 0
        print("Contour count is " + str(len(contours)))
        new_imgs = []
        for c in contours:
            # Returns the location and width,height for every contour
            x, y, w, h = cv2.boundingRect(c)
            # If the box height is greater then 20, widht is >80, then only save it as a box in "cropped/" folder.
            if (w > width / 5 and h > height / 5 and w < width and h < height):
                idx += 1
                new_img = img[y:y + h, x:x + w]
                new_imgs.append(new_img)

        if new_imgs == []:
            new_imgs.append(img)

        return new_imgs[0]

    def request_callback(self):
        curr_image = self.img
        crop = self.preprocess_on_fly(curr_image)

        test_img = crop.flatten().reshape(1, 33 * 25)
        test_img = test_img.astype(np.float32)

        ret, results, neighbours, dist = self.knn.findNearest(test_img, 3)

        self.classification_publisher.publish(ret)



    def train(self):

        with open('./img/train.txt', 'rb') as f:
            reader = csv.reader(f)
            lines = list(reader)
        for i in range(len(lines)):  # len(lines)
            img = cv2.imread("./imgs/" + lines[i][0] + ".png", 0)
            height, width = img.shape

            (thresh, img_bin) = cv2.threshold(img, 100, 255,
                                              cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # Thresholding the image
            img_bin = 255 - img_bin  # Invert the image


            # Defining a kernel length
            kernel_length = np.array(img).shape[1] // 40

            # A verticle kernel of (1 X kernel_length), which will detect all the verticle lines from the image.
            verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
            # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
            hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
            # A kernel of (3 X 3) ones.
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            # Morphological operation to detect verticle lines from an image
            img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=3)
            verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3)

             # Morphological operation to detect horizontal lines from an image
            img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)
            horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)
            # Weighting parameters, this will decide the quantity of an image to be added to make a new image.
            alpha = 0.5
            beta = 1.0 - alpha
            # This function helps to add two image with specific weight parameter to get a third image as summation of two image.
            img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
            img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
            (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            # For Debugging
            # Enable this line to see verticle and horizontal lines in the image which is used to find boxes
            # Find contours for image, which will detect all the boxes
            im2, contours, hierarchy = cv2.findContours(
                img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # Sort all the contours by top to bottom.
            (contours, boundingBoxes) = self.sort_contours(contours, method="top-to-bottom")


            idx = 0
            print("Contour count is " + str(len(contours)))
            for c in contours:
                # Returns the location and width,height for every contour
                x, y, w, h = cv2.boundingRect(c)
                # If the box height is greater then 20, widht is >80, then only save it as a box in "cropped/" folder.
                if (w > width / 5 and h > height / 5 and w < width and h < height):
                    idx += 1
                    new_img = img[y:y + h, x:x + w]
                    cv2.imwrite('./imgs_mod/' + lines[i][0] + '_' + str(idx) + '.png', new_img)


        with open('./imgs/train.txt', 'rb') as f:
            reader = csv.reader(f)
            lines = list(reader)

        train = []
        for i in range(len(lines)):
            try:
                if lines[i][1] != '0':
                    train.append(np.array(cv2.resize(cv2.imread("./imgs_mod/" + lines[i][0] + "_1.png", 0), (33, 25))))
                else:
                    train.append(np.array(cv2.resize(cv2.imread("./imgs/" + lines[i][0] + ".png", 0), (33, 25))))

            except:
                a = cv2.imread("./imgs_mod/" + lines[i][0] + "_1.png", 0)
                print(lines[i])
                print(a.shape)

        train = np.array(train)
        # this line reads in all images listed in the file in GRAYSCALE, and resizes them to 33x25 pixels
        # here we reshape each image into a long vector and ensure the data type is a float (which is what KNN wants)
        train_data = train.flatten().reshape(len(lines), 33 * 25)
        train_data = train_data.astype(np.float32)

        # read in training labels
        train_labels = np.array([np.int32(lines[i][1]) for i in range(len(lines))])

        ### Train classifier
        self.knn = cv2.ml.KNearest_create()
        self.knn.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)


        return


