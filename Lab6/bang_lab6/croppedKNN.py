#!/usr/bin/env python

import cv2
import sys
import argparse
import csv
import numpy as np


blur_size = 3
lower = np.array([0,80,0],np.uint8)
upper = np.array([180,255,255],np.uint8)
val = np.array([0,80,0],np.uint8)
error = np.array([13,50,50],np.uint8)
morphOpSize = 5
maxObjects = 1
minObjectArea = 1500
name = "Object!"

#Color or grey classifier.
colorBool = True

#Number of neighbors to consider
K = 3	

def _mouseEvent(event, x, y, flags, param):
    # The mouse event is connected to the "Original Image Window" and triggers the event when the user click on the image.
    # This event calculates the lower and upper bounds to filter the colors of the image
    global img_hsv
    global lower
    global upper
    global error
    
    if event == cv2.EVENT_LBUTTONDOWN:
        #lower[0] = img_hsv[y,x,0]
        #upper[0] = img_hsv[y,x,0]
        val = img_hsv[y,x,:]
        #upper = img_hsv[y,x,:]     

        #lower = cv2.subtract(lower,error)
        #upper = cv2.add(upper,error)
        print("Hue color:") 
        print("- pixel val:")
        print(val)
        #print("- Upper bound: ")
        #print(upper)

def morphOps(binaryMatrix, kernelSize):
    # Morphological operations (open and close) used to reduce noise in the acquired image.
    kernel = np.ones((kernelSize,kernelSize), np.uint8)
    tempFix = cv2.morphologyEx(binaryMatrix,cv2.MORPH_CLOSE, kernel)   # Fill in holes
    fix = tempFix#cv2.morphologyEx(tempFix,cv2.MORPH_OPEN, kernel)             # Get rid of noise
    return fix

def drawCOM(frame, x, y, name):
    cv2.circle(frame,(x,y),5,(0,255,0))
    cv2.putText(frame,name,(x-30,y-25),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),2)

def findObjects(binaryMatrix):
    #Finds the location of the desired object in the image.
    output = []
    trash, contours, hierarchy = cv2.findContours(binaryMatrix, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Contours the image to find blobs of the same color   
    cont = sorted(contours, key = cv2.contourArea, reverse = True)[:maxObjects]                   # Sorts the blobs by size (Largest to smallest) 

    global imgTrack
    global img_Crop
    global res
    # Find the center of mass of the blob if there are any
    if hierarchy is not None:
        for i in range (0,len(cont)):
            M = cv2.moments(cont[i])
            if M['m00'] > minObjectArea:                                   # Check if the total area of the contour is large enough to care about!
                x,y,w,h = cv2.boundingRect(cont[i])
                rect = cv2.minAreaRect(cont[0])
                w = int(rect[1][0])
                x = int(M['m10']/M['m00'])
                y = int(M['m01']/M['m00'])
                #cv2.drawContours(imgTrack, cont[i], -1, (255,0,0), 3) # Draws the contour.
                img_Crop = img_hsv[y-h/2:y+h/2,x-w/2:x+w/2].copy()
                cv2.rectangle(imgTrack,(x-w/2,y-h/2),(x+w/2,y+h/2),(0,255,0),2)
                drawCOM(imgTrack,x,y,name)
                if output == []:
                    output = [[x,w]]
                else:
                    output.append[[x,w]]

    if len(output) is 0: #We didn't find anything to track take whole image in (blank wall most likely)
    	img_Crop = imgTrack.copy()

    return output


### Load training images and labels
with open('./imgs/train.txt', 'rb') as train:
    reader_train = csv.reader(train)
    lines_train = list(reader_train)

# read in training labels
train_labels = np.array([np.int32(lines_train[i][1]) for i in range(len(lines_train))])

if(__debug__):
	Title_images = 'Original Image'
	Title_mask = 'Image Mask'
	Title_result = 'Image Passed to KNN'
	cv2.namedWindow( Title_images, cv2.WINDOW_AUTOSIZE )
	cv2.setMouseCallback(Title_images,_mouseEvent)

train_list = []
for i in range(0,len(lines_train)):
	img_bgr = cv2.imread("./imgs/"+lines_train[i][0]+".png",1)
	imgTrack = img_bgr.copy()
	img_blur = cv2.GaussianBlur(img_bgr,(blur_size,blur_size),0)
	img_hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(img_hsv, lower, upper)
	res = cv2.bitwise_and(imgTrack,imgTrack,mask = mask)
	imgMorphOps = morphOps(mask, morphOpSize)
	centers = findObjects(imgMorphOps)
	img_lil = cv2.resize(img_Crop,(40,40))
	img_gray = cv2.cvtColor(img_lil, cv2.COLOR_BGR2GRAY)
	
	if(colorBool):
		train_list.append(np.array(img_lil))
	else:
		train_list.append(np.array(img_gray))
	if(__debug__):
		print(train_labels[i])
		cv2.imshow(Title_images,img_bgr)
		cv2.imshow(Title_mask,imgMorphOps)
		if(colorBool):
			cv2.imshow(Title_result,img_lil)
		else:
			cv2.imshow(Title_result,img_gray)
		k = cv2.waitKey()
		if k==27:    # Esc key to stop
			break

train2= np.asarray(train_list)
if(colorBool):
	train_data = train2.flatten().reshape(len(lines_train), 40*40*3)
else:
	train_data = train2.flatten().reshape(len(lines_train), 40*40)
train_data = train_data.astype(np.float32)


### Train classifier
knn = cv2.ml.KNearest_create()
knn.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)


### Run test images
with open('./rodrigoImages/extra_train.txt', 'rb') as f:
    reader = csv.reader(f)
    lines = list(reader)

correct = 0.0
confusion_matrix = np.zeros((6,6))
error_count = 0
for i in range(len(lines)):
	print("current line number is " + str(i))
	print(str(lines[i]))
	try:
		img_bgr = cv2.imread("./rodrigoImages/"+lines[i][0]+".png",1)
		if img_bgr is None: print("NOT ABLE TO READ IMAGE!!!!")
		imgTrack = img_bgr.copy()
		img_blur = cv2.GaussianBlur(img_bgr,(blur_size,blur_size),0)
		img_hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
		mask = cv2.inRange(img_hsv, lower, upper)
		res = cv2.bitwise_and(imgTrack,imgTrack,mask = mask)
		imgMorphOps = morphOps(mask, morphOpSize)
		centers = findObjects(imgMorphOps)
		if img_Crop is None: print("Cropped image does not exist!!!")
		img_lil = cv2.resize(img_Crop,(40,40))
		img_gray = cv2.cvtColor(img_lil, cv2.COLOR_BGR2GRAY)

		if(colorBool):	
			test_img = np.asarray(img_lil)
			test_img = test_img.flatten().reshape(1, 40*40*3)
		else:
			test_img = np.asarray(img_gray)
			test_img = test_img.flatten().reshape(1, 40*40)
		test_img = test_img.astype(np.float32)
		test_label = np.int32(lines[i][1])

		ret, results, neighbours, dist = knn.findNearest(test_img, K)

		if test_label == ret:
			print(lines[i][0], " Correct, ", ret)
			correct += 1
			confusion_matrix[np.int32(ret)][np.int32(ret)] += 1
		else:
			confusion_matrix[test_label][np.int32(ret)] += 1
        
			print(lines[i][0], " Wrong, ", test_label, " classified as ", ret)
			print "\tneighbours: ", neighbours

		if(__debug__):
			cv2.imshow(Title_images,img_bgr)
			cv2.imshow(Title_mask,imgMorphOps)
			if(colorBool):
				cv2.imshow(Title_result,img_lil)
			else:
				cv2.imshow(Title_result,img_gray)
			k = cv2.waitKey()
			if k==27:    # Esc key to stop
				break
	except:
		correct += 1
		error_count += 1
	
print("Total accuracy: ", correct/len(lines))
print(confusion_matrix)
print("Error count: " + str(error_count))
cv2.destroyAllWindows()
