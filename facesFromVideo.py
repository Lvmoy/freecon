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
params["face"] = True
params["face_detector"] = 2
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
        # faceRectangles = [
        #     op.Rectangle(0.,0.,480.,480.)
        #     op.Rectangle(330.119385, 277.532715, 48.717274, 48.717274),
        #     op.Rectangle(24.036991, 267.918793, 65.175171, 65.175171),
        #     op.Rectangle(151.803436, 32.477852, 108.295761, 108.295761),
        # ]

        faceRectangles = [
            op.Rectangle(0.,0.,480.,480.)
        ]

        # Create new datum
        datum = op.Datum()
        datum.cvInputData = imageToProcess
        datum.faceRectangles = faceRectangles

        # Process and display image
        opWrapper.emplaceAndPop([datum])
        cv2.namedWindow("OpenPose 1.4.0 - Tutorial Python API", cv2.WINDOW_NORMAL)
        print("Face keypoints: \n" + str(datum.faceKeypoints))
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
