
import datetime
import numpy as np
import cv2

from random import choice
from numpy import load
from numpy import asarray
from numpy import expand_dims
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
import tensorflow as tf
from sklearn.svm import SVC
from matplotlib import pyplot
from PIL import Image

from keras.models import load_model

import argparse
import sys
import os

# Parameters
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
IMG_WIDTH = 416
IMG_HEIGHT = 416

# Default colors
COLOR_BLUE = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_YELLOW = (0, 255, 255)


class YoloDark():
    def __init__(self,boundingbox=True, port=0, modelCfg="yolo/cfg/yolov3-face.cfg", modelWeight= "yolo/weights/yolov3-wider_16000.weights" ):
        self.modelCfg=modelCfg
        self.modelWeight=modelWeight
        self.port=port#"test/sample.mp4"
        self.boundingbox=boundingbox

        

    def yoloInit(self):
        self.net = cv2.dnn.readNetFromDarknet(self.modelCfg, self.modelWeight)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return self.net

    def yoloProcess(self,net,frame):
        # Create a 4D blob from a frame.
        blob = cv2.dnn.blobFromImage(frame, 1/255 , (IMG_WIDTH, IMG_HEIGHT),
                         [0, 0, 0], 1, crop=False)
        # Sets the input to the network
        net.setInput(blob)
        # Runs the forward pass to get output of the output layers
        outs = net.forward(self.get_outputs_names(net))
        # Remove the bounding boxes with low confidence
        faces = self.post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)
        return faces
    
    def faces(self,frame):
        return self.yoloProcess(self.net,frame)


    def run(self): #main
        net= self.yoloInit()
        cap = cv2.VideoCapture(self.port)
        i=0
        while True:
            has_frame, frame = cap.read()
            if(i==10):
                if not has_frame:
                    print('Done processing')
                    cv2.waitKey(1000)
                    break
                faces= self.yoloProcess(net,frame)
                for face in faces:
                    x1=face[0]
                    y1=face[1]
                    x2=face[0]+face[2]
                    y2=face[1]+face[3]
                    face = pixels[y1:y2, x1:x2]
                    image = Image.fromarray(face)
                    image = image.resize((160,160))
                    image = np.asarray(image)
                cv2.imshow("Cam", frame)
                i=0
            i=i+1

            key = cv2.waitKey(1)
            if key == 27 or key == ord('q'):
                print('[i] ==> Interrupted by user!')
                break
        cap.release()
        cv2.destroyAllWindows()

    def get_outputs_names(self,net):
        # Get the names of all the layers in the network
        layers_names = net.getLayerNames()
        # Get the names of the output layers, i.e. the layers with unconnected
        # outputs
        return [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # Draw the predicted bounding box
    def draw_predict(self, frame, conf, left, top, right, bottom):
        # Draw a bounding box.

        # cv2.rectangle(frame, (left-1, top-5), (right+1, bottom+5), COLOR_YELLOW, 2)
        pass
        # text = '{:.2f}'.format(conf)

        # Display the label at the top of the bounding box
        # label_size, base_line = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        #
        # top = max(top, label_size[1])
        # cv2.putText(frame, text, (left, top - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,COLOR_WHITE, 1)

    def post_process(self, frame, outs, conf_threshold, nms_threshold):
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        # Scan through all the bounding boxes output from the network and keep only
        # the ones with high confidence scores. Assign the box's class label as the
        # class with the highest score.
        confidences = []
        boxes = []
        final_boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > conf_threshold:
                    center_x = int(detection[0] * frame_width)
                    center_y = int(detection[1] * frame_height)
                    width = int(detection[2] * frame_width)
                    height = int(detection[3] * frame_height)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        # overlapping boxes with lower confidences.
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold,
                                   nms_threshold)
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            final_boxes.append(box)
            left, top, right, bottom = self.refined_box(left, top, width, height)
            # draw_predict(frame, confidences[i], left, top, left + width,
            #              top + height)
            if self.boundingbox:
                self.draw_predict(frame, confidences[i], left, top, right, bottom)
        return final_boxes

    def refined_box(self, left, top, width, height):
        right = left + width
        bottom = top + height

        original_vert_height = bottom - top
        top = int(top + original_vert_height * 0.15)
        bottom = int(bottom - original_vert_height * 0.05)

        margin = ((bottom - top) - (right - left)) // 2
        left = left - margin if (bottom - top - right + left) % 2 == 0 else left - margin - 1

        right = right + margin

        return left, top, right, bottom
        
