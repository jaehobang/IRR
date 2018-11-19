#!/usr/bin/env python

import cv2
import csv
import numpy as np
from sklearn.neural_network import MLPClassifier


class ImageClassifier:


    def __init__(self):
        self.dnn = self.dnn()


    def train(self):
        with open('./imgs/train.txt', 'rb') as f:
            reader = csv.reader(f)
            lines = list(reader)

        # this line reads in all images listed in the file in GRAYSCALE, and resizes them to 33x25 pixels
        train = np.array(
            [np.array(cv2.resize(cv2.imread("./imgs/" + lines[i][0] + ".png", 0), (33, 25))) for i in
             range(len(lines))])

        # here we reshape each image into a long vector and ensure the data type is a float (which is what KNN wants)
        train_data = train.flatten().reshape(len(lines), 33 * 25)
        train_data = train_data.astype(np.float32)

        # read in training labels
        train_labels = np.array([np.int32(lines[i][1]) for i in range(len(lines))])

        #Train the neural network
        self.dnn_train(train_data, train_labels)

    def test(self):
        ### Run test images

        with open('./imgs/test.txt', 'rb') as f:
            reader = csv.reader(f)
            lines = list(reader)

        test = np.array(
            [np.array(cv2.resize(cv2.imread("./imgs/" + lines[i][0] + ".png", 0), (33, 25))) for i in
             range(len(lines))])

        # here we reshape each image into a long vector and ensure the data type is a float (which is what KNN wants)
        test_data = test.flatten().reshape(len(lines), 33 * 25)
        test_data = test_data.astype(np.float32)

        test_labels = np.array([np.int32(lines[i][1]) for i in range(len(lines))])

        print("Accuracy is " + self.dnn_score(test_data, test_labels))

        return

    def main(self):
        self.train()
        self.test()


    def predict(self, test_data):
        #assume it is already flattened and type is changed to np.float32
        return self.dnn_predict(test_data)



    def dnn_init(self):
        dnn = MLPClassifier(solver='lbfgs', alpha=1e-5,
                              hidden_layer_sizes = (5, 2), random_state = 1)
        return dnn

    def dnn_train(self, train_data, train_label):
        if __debug__: print("Training dnn...")
        self.dnn.fit(train_data, train_label)
        if __debug__: print("Done")

    def dnn_predict(self, test_data):
        return self.dnn.predict(test_data)

    def dnn_score(self, test_data, test_label):
        if __debug__: print("Computing dnn score...")
        return self.dnn.score(test_data, test_label)
        if __debug__: print("Done")


if __name__ == "__main__":
    image_classifier = ImageClassifier()
    image_classifier.main()
