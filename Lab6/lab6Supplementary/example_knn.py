#!/usr/bin/env python

import cv2
import csv
import numpy as np

### Load training images and labels




with open('./imgs/train.txt', 'rb') as f:
    reader = csv.reader(f)
    lines = list(reader)



train = []
for i in range(len(lines)):
    try:
        if lines[i][1] != '0':
            train.append( np.array(cv2.resize(cv2.imread("./imgs_mod/" + lines[i][0] + "_1.png", 0), (33, 25))) )
        else:
            train.append( np.array(cv2.resize(cv2.imread("./imgs/" + lines[i][0] + ".png", 0), (33, 25))) )

    except:
        a = cv2.imread("./imgs_mod/" + lines[i][0] + "_1.png", 0)
        print(lines[i])
        print(a.shape)

train = np.array(train)
# this line reads in all images listed in the file in GRAYSCALE, and resizes them to 33x25 pixels
# here we reshape each image into a long vector and ensure the data type is a float (which is what KNN wants)
train_data = train.flatten().reshape(len(lines), 33*25)
train_data = train_data.astype(np.float32)

# read in training labels
train_labels = np.array([np.int32(lines[i][1]) for i in range(len(lines))])


### Train classifier
knn = cv2.ml.KNearest_create()
knn.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)


### Run test images
with open('./imgs/test.txt', 'rb') as f:
    reader = csv.reader(f)
    lines = list(reader)

correct = 0.0
confusion_matrix = np.zeros((6,6))

for i in range(len(lines)):
    if lines[i][1] != '0':
        test_img = np.array(cv2.resize(cv2.imread("./imgs_mod/" + lines[i][0] + "_1.png", 0), (33, 25)))
    else:
        test_img = np.array(cv2.resize(cv2.imread("./imgs/" + lines[i][0] + ".png", 0), (33, 25)))

    test_img = test_img.flatten().reshape(1, 33*25)
    test_img = test_img.astype(np.float32)

    test_label = np.int32(lines[i][1])

    ret, results, neighbours, dist = knn.findNearest(test_img, 3)

    if test_label == ret:
        print(lines[i][0], " Correct, ", ret)
        correct += 1
        confusion_matrix[np.int32(ret)][np.int32(ret)] += 1
    else:
        confusion_matrix[test_label][np.int32(ret)] += 1
        
        print(lines[i][0], " Wrong, ", test_label, " classified as ", ret)
        print "\tneighbours: ", neighbours
        print "\tdistances: ", dist



print("\n\nTotal accuracy: ", correct/len(lines))
print(confusion_matrix)
