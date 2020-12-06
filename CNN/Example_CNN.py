import math
import numpy as np
import h5py
import tensorflow as tf

import scipy

from scipy import ndimage
from PIL import Image

from tensorflow.python.framework import ops
import matplotlib.pyplot as plt

# Imports the Helper Functions like Model_CNN() etc. from cnn_functions.py in same directory.
from cnn_functions import *


train_dataset = h5py.File('train_signs.h5', "r")
X_train_orig = np.array(train_dataset["train_set_x"][:])        # train set features
Y_train_orig = np.array(train_dataset["train_set_y"][:])        # train set labels

test_dataset = h5py.File('test_signs.h5', "r")
X_test_orig = np.array(test_dataset["test_set_x"][:])           # test set features
Y_test_orig = np.array(test_dataset["test_set_y"][:])           # test set labels

classes = np.array(test_dataset["list_classes"][:])             # the list of classes
    
Y_train_orig = Y_train_orig.reshape((1, Y_train_orig.shape[0]))
Y_test_orig = Y_test_orig.reshape((1, Y_test_orig.shape[0]))


# Example of a picture
index = 6
plt.subplot(1, 2, 1)
plt.imshow(X_train_orig[index])

print ("Plotting the Image " + str(np.squeeze(Y_train_orig[:, index])) + "\n")


X_train = X_train_orig/255.
X_test = X_test_orig/255.
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T
print ("Number of training examples = " + str(X_train.shape[0]))
print ("Number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape) + "\n")
conv_layers = {}


X, Y = create_placeholders(64, 64, 3, 6)

plt.subplot(1, 2, 2)
_, _, parameters = Model_CNN(X_train, Y_train, X_test, Y_test)
