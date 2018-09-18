#Importing the Libraries
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import glob
import re
import os
import cv2
from random import shuffle

#Libraries required for traning convolutional neural network
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

image_dir = 'D:/Data_Science_Work/real-time-object-detection/dataSet/'
MAX_NUM_IMAGES_PER_CLASS = 20
IMG_SIZE = 50
LR = 0.0001

MODEL_NAME = 'face_detect-{}-{}.model2'.format(LR, '2conv-basic') # just so we remember which saved model is which, sizes must match


def label_img(img_path):
    label = img_path.split('_')[1]
    if label == "Manoranjan" :return [1,0]
    else : return [0,1]
    
def create_train_data():
    training_data = []
    for img in os.listdir(image_dir):
        label = label_img(img)
        path = os.path.join(image_dir,img)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img),np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data
    
def train_model():
    
    convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')
    
    convnet = conv_2d(convnet, 32, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)
    
    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)
    convnet = dropout(convnet, 0.4)
    
    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)    
    
    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.4)
    
    convnet = fully_connected(convnet, 2, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
    
    model = tflearn.DNN(convnet, tensorboard_dir='log')    
    
    if os.path.exists('{}.meta'.format(MODEL_NAME)):
        model.load(MODEL_NAME)
        print('model loaded!')
        
    train_data = np.load('train_data.npy')
    train = train_data[:-150]
    test = train_data[-150:]
    
    X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
    Y = [i[1] for i in train] 
    test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
    test_y = [i[1] for i in test]
    early_stopping_cb = EarlyStoppingCallback(val_acc_thresh=0.9)    
    model.fit({'input': X}, {'targets': Y}, n_epoch=4, validation_set=({'input': test_x}, {'targets': test_y}),
              show_metric=True, run_id=MODEL_NAME, shuffle=True, batch_size= 20)    
    
    model.save(MODEL_NAME)
    
def main():
    create_train_data()
    train_model()


class EarlyStoppingCallback(tflearn.callbacks.Callback):
    def __init__(self, val_acc_thresh):
        """ Note: We are free to define our init function however we please. """
        self.val_acc_thresh = val_acc_thresh
    
    def on_epoch_end(self, training_state):
        """ """
        # Apparently this can happen.
        if training_state.val_acc is None: return
        if training_state.val_acc > self.val_acc_thresh:
            raise StopIteration

if __name__ == "__main__":
    main()
