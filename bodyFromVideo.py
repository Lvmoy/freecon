import cv2
import numpy as np
import os
import sys
import time
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-v", "--cudaversion", type=int, choices=[8,9])
args = parser.parse_args()

dir_path = ""
model_path = ""
if args.cudaversion == 9:
    dir_path = '/home/li/Work/openpose-cuda9.0/build/python/'
    model_path = "/home/li/Work/openpose-cuda9.0/models/"

else:
    dir_path = '/home/li/Work/openpose-cuda8.0/build/python/'
    model_path = "/home/li/Work/openpose-cuda8.0/models/"
#dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)
sys.path.append(dir_path)
from openpose import pyopenpose as op

cam = cv2.VideoCapture(0)
print(cam.isOpened())
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1980)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280)


params = dict()
params["model_folder"] = model_path
params["face"] = True
params["hand"] = True


try:
    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
    
    while(True):
        rec, imageToProcess = cam.read()
        if not rec :
            break

        start = time.time()


        # Create new datum
        datum = op.Datum()
        datum.cvInputData = imageToProcess
        # Process and display image
        opWrapper.emplaceAndPop([datum])

        cv2.namedWindow("OpenPose 1.4.0 - Tutorial Python API", cv2.WINDOW_NORMAL)
        print("Body keypoints: \n" + str(datum.poseKeypoints))
        print("Face keypoints: \n" + str(datum.faceKeypoints))
        print("Left hand keypoints: \n" + str(datum.handKeypoints[0]))
        print("Right hand keypoints: \n" + str(datum.handKeypoints[1]))
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
