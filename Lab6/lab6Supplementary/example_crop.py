#!/usr/bin/env python

import cv2
import csv
import numpy as np
import os







class Preprocess:



    def __init__(self):
        self.train_data = None
        self.train_labels = None
        self.img = None
        self.x = 50
        self.y = 50

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


    def detect_box(self, dir):

        with open(dir, 'rb') as f:
            reader = csv.reader(f)
            lines = list(reader)
        for i in range(len(lines)):  # len(lines)
            img = cv2.imread("./imgs/" + lines[i][0] + ".png", 0)
            height, width = img.shape

            (thresh, img_bin) = cv2.threshold(img, 100, 255,
                                              cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # Thresholding the image
            img_bin = 255 - img_bin  # Invert the image

            if __debug__:
                cv2.imshow("Image_bin.jpg", img_bin)
                cv2.waitKey(0) & 0xFF
                cv2.destroyAllWindows()

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

            if __debug__:
                cv2.imshow("verticle_lines.jpg", verticle_lines_img)
                cv2.waitKey(0) & 0xFF
                cv2.destroyAllWindows()
            # Morphological operation to detect horizontal lines from an image
            img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)
            horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)
            if __debug__:
                cv2.imshow("horizontal_lines.jpg", horizontal_lines_img)
                cv2.waitKey(0) & 0xFF
                cv2.destroyAllWindows()
            # Weighting parameters, this will decide the quantity of an image to be added to make a new image.
            alpha = 0.5
            beta = 1.0 - alpha
            # This function helps to add two image with specific weight parameter to get a third image as summation of two image.
            img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
            img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
            (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            # For Debugging
            # Enable this line to see verticle and horizontal lines in the image which is used to find boxes
            if __debug__:
                cv2.imshow("img_final_bin.jpg", img_final_bin)
                cv2.waitKey(0) & 0xFF
                cv2.destroyAllWindows()
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



    def preprocess(self):
        with open('./imgs/train.txt', 'rb') as f:
            reader = csv.reader(f)
            lines = list(reader)


        for i in range(5): #len(lines)
            img = cv2.imread("./imgs/" + lines[i][0] + ".png", cv2.IMREAD_COLOR)

            imageHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            maskRed = cv2.inRange(imageHSV, np.array([50, 20, 100]), np.array([179, 255, 255]));

            imageSeg = cv2.bitwise_and(imageHSV, imageHSV, mask=maskRed)
            imageSegBRG = cv2.cvtColor(imageSeg, cv2.COLOR_HSV2BGR)

            cv2.imshow('image', imageSegBRG)
            cv2.waitKey(0) & 0xFF
            cv2.destroyAllWindows()

            im2, contours, hierarchy = cv2.findContours(maskRed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnt = contours[0]
            x, y, w, h = cv2.boundingRect(cnt)

            crop = img[y:y + h, x:x + w]
            cv2.imwrite('./imgs_mod/' + lines[i][0] + '.png', crop)


if __name__ == "__main__":
    if not os.path.isdir("./img_mod"):
        os.mkdir("img_mod")
    preprocess = Preprocess()
    preprocess.detect_box('./imgs/test.txt')
