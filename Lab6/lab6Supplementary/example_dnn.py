#!/usr/bin/env python

import cv2
import csv
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten


class ImageClassifier:


    def __init__(self):
        self.dnn = self.dnn_init()
        self.pca = self.pca_init()
        self.train_data = None
        self.train_labels = None
        self.test_data = None
        self.test_labels = None
        self.cnn = None
        self.x = 28
        self.y = 28

    def load_train_data(self):
        with open('./imgs/train.txt', 'rb') as f:
            reader = csv.reader(f)
            lines = list(reader)

        # this line reads in all images listed in the file in GRAYSCALE, and resizes them to 33x25 pixels

        train = []
        for i in range(len(lines)):
            try:
                if lines[i][1] != '0':
                    train.append( np.array(cv2.resize(cv2.imread("./imgs_mod/" + lines[i][0] + "_1.png", 0), (self.x, self.y))) )
                else:
                    train.append( np.array(cv2.resize(cv2.imread("./imgs/" + lines[i][0] + ".png", 0), (self.x, self.y))) )

            except:
                a = cv2.imread("./imgs_mod/" + lines[i][0] + "_1.png", 0)
                print(lines[i])
                print(a.shape)

        train = np.array(train)

        self.train_data = train
        # here we reshape each image into a long vector and ensure the data type is a float (which is what KNN wants)
        train_data = train.flatten().reshape(len(lines), self.x * self.y)
        self.train_data = train_data.astype(np.float32)

        # read in training labels
        self.train_labels = np.array([np.int32(lines[i][1]) for i in range(len(lines))])


    def load_test_data(self):
        ### Run test images

        with open('./imgs/test.txt', 'rb') as f:
            reader = csv.reader(f)
            lines = list(reader)

        test = []
        for i in range(len(lines)):
            if lines[i][1] != '0':
                test.append(
                    np.array(cv2.resize(cv2.imread("./imgs_mod/" + lines[i][0] + "_1.png", 0), (self.x, self.y))))
            else:
                test.append(np.array(cv2.resize(cv2.imread("./imgs/" + lines[i][0] + ".png", 0), (self.x, self.y))))

        test = np.array(test)

        """
        test = np.array(
            [np.array(cv2.resize(cv2.imread("./imgs_mod/" + lines[i][0] + "_1.png", 0), (self.x, self.y))) if lines[i][1] != 0 else
             np.array(cv2.resize(cv2.imread("./imgs/" + lines[i][0] + ".png", 0), (self.x, self.y))) for i in range(len(lines))])

        """


        self.test_data = test
        test_data = test.flatten().reshape(len(lines), self.x * self.y)
        self.test_data = test_data.astype(np.float32)

        # here we reshape each image into a long vector and ensure the data type is a float (which is what KNN wants)

        self.test_labels = np.array([np.int32(lines[i][1]) for i in range(len(lines))])



    def keras_train(self):
        # one-hot encode target column
        self.train_labels = to_categorical(self.train_labels)
        # one-hote encoding for labels
        self.test_labels = to_categorical(self.test_labels)

        # create model
        model = Sequential()
        # add model layers
        model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape = (self.y, self.x, 3)))
        model.add(Conv2D(32, kernel_size=3, activation='relu'))
        model.add(Flatten())
        model.add(Dense(6, activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        model.fit(self.train_data, self.train_labels, validation_data=(self.test_data, self.test_labels), epochs=3)
        self.cnn = model




    def main(self):
        self.load_train_data()
        self.load_test_data()
        #self.keras_train()
        self.dnn_train()
        self.dnn_score()


    def predict(self, test_data):
        #assume it is already flattened and type is changed to np.float32
        return self.dnn_predict(test_data)


    def pca_init(self):
        pca = PCA()
        return pca


    def dnn_init(self):
        dnn = MLPClassifier(solver='lbfgs', alpha=1e-5,
                            hidden_layer_sizes = (15, 2),
                            random_state = 1,
                            batch_size='auto',
                            max_iter=200000,
                            shuffle=True,
                            tol=0.000001
                            )
        return dnn

    def dnn_train(self):

        if __debug__: print("Training dnn...")
        self.dnn.fit(self.train_data, self.train_labels)
        if __debug__: print("Done")

    def dnn_predict(self, test_data):
        return self.dnn.predict(test_data)

    def dnn_score(self):
        if __debug__: print("Computing dnn score...")
        print self.dnn.score(self.test_data, self.test_labels)
        if __debug__: print("Done")


if __name__ == "__main__":
    image_classifier = ImageClassifier()
    image_classifier.main()
