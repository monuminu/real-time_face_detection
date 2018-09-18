# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import tensorflow as tf
from tensorflow.python.platform import gfile
import glob
import re
import os

image_dir = 'D:/Data_Science_Work/real-time-object-detection/dataSet/'
MAX_NUM_IMAGES_PER_CLASS = 20
IMG_SIZE = 50
LR = 0.0001

#Libraries required for traning convolutional neural network
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

classes = ['Manoranjan', 'Sanghamitra']
arg_confidence = 0.85

def load_model():
    MODEL_NAME = 'face_detect-{}-{}.model2'.format(LR, '2conv-basic') # just so we remember which saved model is which, sizes must match
    convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')
    
    convnet = conv_2d(convnet, 32, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)
    
    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)
    
    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)    
    
    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)
    
    convnet = fully_connected(convnet, 2, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
    
    model = tflearn.DNN(convnet, tensorboard_dir='log')    
    if os.path.exists('{}.meta'.format(MODEL_NAME)):
        model.load(MODEL_NAME)
        print('model loaded!')
    return model


def detect_face(model,img_data):
    img_data = cv2.resize(img_data, (IMG_SIZE,IMG_SIZE))
    data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
    model_out = model.predict([data])[0]
    #print(model_out)
    str_label = "Unknown"
    for i, confidence in enumerate(model_out):
        if confidence > arg_confidence :
            str_label = classes[i]
    return str_label
    
    
def start_process(model):
    # initialize the video stream, allow the cammera sensor to warmup,
    # and initialize the FPS counter
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start() #"http://192.168.0.102:8080/video"
    time.sleep(2.0)
    fps = FPS().start()
    
    #load the required files to face detection
    face_cascade = cv2.CascadeClassifier('D:/Data_Science_Work/real-time-object-detection/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('D:/Data_Science_Work/real-time-object-detection/haarcascade_eye.xml')
    
    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = vs.read()
        #frame = cv2.flip(frame, 1)
        frame = imutils.resize(frame, width=400)
        
        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                         0.007843, (300, 300), 127.5)    
        
        #Converting frame to Gray
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            label = detect_face(model,roi_gray)
            roi_color = frame[y:y+h, x:x+w]
            cv2.putText(frame, label, (x-3, y-3),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10,255,255), 2)
            
        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
    
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
    
        # update the FPS counter
        fps.update()    
    
def main():
    model = load_model()
    start_process(model)
    
if __name__ == "__main__":
    main()
