import sys
import time
import cv2
import numpy as np
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

print(dir_path)

sys.path.append(dir_path)
from openpose import pyopenpose as op

cam = cv2.VideoCapture(0, cv2.CAP_V4L)
print(cam.isOpened())

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1980)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280)

params = dict()
params["model_folder"] = model_path
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
        print('video radio:', imageToProcess.shape)
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
        cv2.namedWindow("Pose", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Pose-Heat", cv2.WINDOW_NORMAL)
        #cv2.namedWindow("original ", cv2.WINDOW_NORMAL)

        num_maps = heatmaps.shape[0]
        print(num_maps)
        right_heatmap = heatmaps[4, :, :].copy()
        left_heatmap = heatmaps[7, :, :].copy()

        right_blur = right_heatmap / 255.
        right_blur = cv2.GaussianBlur(right_heatmap, ksize=(3,3),sigmaX=0, sigmaY=0)

        print(right_blur)
        print('before map:', right_blur.shape)
        cv2.imshow("Pose-Heat", right_blur)

        heatmap = right_heatmap + left_heatmap
        
        #print(heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        #print('after colormap:', heatmap.shape)
        #print('output:', outputImageF.shape)
        combined = cv2.addWeighted(outputImageF, 0.5, heatmap, 0.5, 0)
        cv2.imshow("Pose", combined)
        #cv2.imshow("origianl", imageToProcess)
        realend = time.time()
        print("cal and show run  time : " , str(realend - start))

        #cv2.waitKey(0)
        if(cv2.waitKey(1)&0xff == ord('q')):
            break
    
except Exception as e:
    # print(e)
    sys.exit(-1)


cv2.waitKey(1)
cam.release()
cv2.destroyAllWindows()


#import numpy as np
#mylist = largerzero(np.zeros((4,5), np.int32))