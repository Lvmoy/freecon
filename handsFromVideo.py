import cv2
import numpy as np
import os
import sys
import time

#dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = '/home/li/Work/openpose/build/python/'
print(dir_path)
sys.path.append(dir_path)
from openpose import pyopenpose as op

cam = cv2.VideoCapture(0, cv2.CAP_V4L)
print(cam.isOpened())
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1980)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280)

params = dict()
params["model_folder"] = "/home/li/Work/openpose/models/"
params["hand"] = True
params["hand_detector"] = 2
params["body"] = 0


try:
    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
    
    while(True):
        start = time.time()
        rec, imageToProcess = cam.read()
        if not rec :
            break

            # Read image and face rectangle locations
        #imageToProcess = cv2.imread(args[0].image_path)
        print(imageToProcess.shape)
        height = imageToProcess.shape[0]
        width = imageToProcess.shape[1]
        handRectangles = [
            # Left/Right hands person 0
            [
            op.Rectangle(0., 0., height, height),
            op.Rectangle(width/2., 0., height, height),
            ],
            # Left/Right hands person 1
            [
            op.Rectangle(0., 0., height, height),
            op.Rectangle(width/2, 0., height, height),
            ]
        ]

        # Create new datum
        datum = op.Datum()
        datum.cvInputData = imageToProcess
        datum.handRectangles = handRectangles

        # Process and display image
        opWrapper.emplaceAndPop([datum])
        end = time.time()
        print("run  time : " , str(end - start))
        cv2.namedWindow("OpenPose 1.4.0 - Tutorial Python API", cv2.WINDOW_NORMAL)
        #print("Left hand keypoints: \n" + str(datum.handKeypoints[0]))
        #print("Right hand keypoints: \n" + str(datum.handKeypoints[1]))
        print("Left hand 0 keypoints: \n" + str(datum.handKeypoints[0][0][0]))
        print(datum.handKeypoints[0].shape)
        cv2.imshow("OpenPose 1.4.0 - Tutorial Python API", datum.cvOutputData)
        realend = time.time()
        print("cal and show run  time : " , str(realend - start))


        #cv2.imshow("hands_video", img)
        if(cv2.waitKey(1)&0xff == ord('q')):
            break
    
except Exception as e:
    # print(e)
    sys.exit(-1)


cv2.waitKey(1)
cam.release()
cv2.destroyAllWindows()
