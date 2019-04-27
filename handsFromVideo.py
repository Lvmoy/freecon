import cv2
import numpy as np
import os
import sys

#dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = '/home/li/Work/openpose/build/python/'
print(dir_path)
sys.path.append(dir_path)
from openpose import pyopenpose as op

cam = cv2.VideoCapture(0)
print(cam.isOpened())


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
            op.Rectangle(0., 0., width/4., width/4.),
            op.Rectangle(width/4., 0., width/2., width/2.),
            ],
            # Left/Right hands person 1
            [
            op.Rectangle(width/2., 0., width*0.75, width*0.75),
            op.Rectangle(width*0.75, 0, width, width),
            ]
        ]

        # Create new datum
        datum = op.Datum()
        datum.cvInputData = imageToProcess
        datum.handRectangles = handRectangles

        # Process and display image
        opWrapper.emplaceAndPop([datum])
        cv2.namedWindow("OpenPose 1.4.0 - Tutorial Python API", cv2.WINDOW_NORMAL)
        print("Left hand keypoints: \n" + str(datum.handKeypoints[0]))
        print("Right hand keypoints: \n" + str(datum.handKeypoints[1]))
        cv2.imshow("OpenPose 1.4.0 - Tutorial Python API", datum.cvOutputData)
        #cv2.waitKey(0)


        #cv2.imshow("hands_video", img)
        if(cv2.waitKey(1)&0xff == ord('q')):
            break
    
except Exception as e:
    # print(e)
    sys.exit(-1)


cv2.waitKey(1)
cam.release()
cv2.destroyAllWindows()
