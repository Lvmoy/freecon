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
params["heatmaps_add_parts"] = True
params["heatmaps_add_bkg"] = True
params["heatmaps_add_PAFs"] = True
params["heatmaps_scale"] = 2


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
       
        # Create new datum
        datum = op.Datum()
        datum.cvInputData = imageToProcess
        # Process and display image
        opWrapper.emplaceAndPop([datum])

        outputImageF = (datum.inputNetData[0].copy())[0,:,:,:] + 0.5
        outputImageF = cv2.merge([outputImageF[0,:,:], outputImageF[1,:,:], outputImageF[2,:,:]])
        outputImageF = (outputImageF*255.).astype(dtype='uint8')
        heatmaps = datum.poseHeatMaps.copy()
        heatmaps = (heatmaps).astype(dtype='uint8')

        end = time.time()
        print("model run  time : " , str(end - start))
        cv2.namedWindow("OpenPose 1.4.0 - Tutorial Python API", cv2.WINDOW_NORMAL)

        num_maps = heatmaps.shape[0]
        heatmap = heatmaps[25, :, :].copy()
        print(heatmaps)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        combined = cv2.addWeighted(outputImageF, 0.5, heatmap, 0.5, 0)
        cv2.imshow("OpenPose 1.4.0 - Tutorial Python API", combined)

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
