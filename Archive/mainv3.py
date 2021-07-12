# USAGE
# py main.py --video video.avi

# import the necessary packages
from imutils.video import FPS
import multiprocessing
import numpy as np
import argparse
import dlib
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from detectFrozen import detect
from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
import time 

def start_tracker(box, label, rgb, inputQueue, outputQueue, refreshRate):
    frame = 1
    start = time.time()
    # construct a dlib rectangle object from the bounding box
    # coordinates and then start the correlation tracker
    t = dlib.correlation_tracker()
    rect = dlib.rectangle(box[0], box[1], box[2], box[3])
    t.start_track(rgb, rect)
    print('a')
    print(time.time()-start)
    # loop indefinitely -- this function will be called as a daemon
    # process so we don't need to worry about joining it
    while True:
        print('b')
        print(time.time()-start)
        # attempt to grab the next frame from the input queue
        rgb = inputQueue.get()
        frame += 1		
        # if there was an entry in our queue, process it
        if rgb is not None:

            
            # update the tracker and grab the position of the tracked
            # object
            buffer = t.update(rgb)
            pos = t.get_position()

            # unpack the position object
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())
            
            
            # add the label + bounding box coordinates to the output
            # queue
            if frame < refreshRate:
                outputQueue.put((label, (startX, startY, endX, endY)))
                print('c')
                print(time.time()-start)
                # print(frame)
            else:
                outputQueue.put(('person', (0,0,0,0)))
                return
    
# helper function to convert bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
def format_boxes(bboxes, image_height, image_width):
    for box in bboxes:
        ymin = int(box[0] * image_height)
        xmin = int(box[1] * image_width)
        ymax = int(box[2] * image_height)
        xmax = int(box[3] * image_width)
        width = xmax - xmin
        height = ymax - ymin
        #box[0], box[1], box[2], box[3] = xmin, ymin, width, height     
        box[0] = xmin
        box[1] = ymin
        box[2] = xmax
        box[3] = ymax
    return bboxes


def fallAlgorithm(startX,startY,endX,endY):
    dx, dy = endX-startX, endY-startY  	# Check the difference
    difference = dy-dx	
    if difference < 0:						
        return (0,0,255)
    else:
        return 	(0,255,0)

def calculateIou(boxA,boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    print(iou)
    return iou

def checkSleep(bedbbox,bedCounter,bb):
    #minX minY maxX maxY
    for i in range (len(bedbbox)):
        if bedbbox[i][0] < bb[0] and bedbbox[i][1] < bb[1]:
            if bedbbox[i][2] > bb[2] and bedbbox[i][3] > bb[3] :
                bedCounter[i] = 1
            else:
                bedCounter[i] = 0
        else:
            bedCounter[i] = 0
        



                

        
def startStream(args):

    # initialize list of beds, each bed will be a series of bounding boxes.
    # xmin ymin xmax ymax
    # 2 & 3 shift left by 5 pxiels
    bedbbox=[[30,80,85,138]]
    bedCounter = [0]
    bedTime = [0]
    
    # to track the frames
    totalFrames = 0

    # initialize the list of class labels MobileNet SSD was trained to
    # detect
    CLASSES = load_classes(parse_data_cfg('data/coco.data')['names'])

    # initialize the video stream and output video writer
    print("[INFO] starting video stream...")
    vs = cv2.VideoCapture(args['video'])
    writer = None

    # start the frames per second throughput estimator
    fps = FPS().start()


    # loop over frames from the video file stream
    while True:
        # grab the next frame from the video file
        (grabbed, frame) = vs.read()
        # check to see if we have reached the end of the video file
        if frame is None:
            break
        else:
            # resize the frame for faster processing and then convert the
            frame = cv2.resize(frame, (128,128))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            # print("Normal")
            # print(frame)
            # print("Gray")
            # print(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))
            # frame from BGR to GRAY ordering 
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Padded resize
            img = letterbox(frame, new_shape=128)[0]

            # Normalize RGB
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
            img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to fp16/fp32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            
            # print(frame.shape)
            # print(np.transpose(image_data[0]).shape)
            # if we are supposed to be writing a video to disk, initialize
            # the writer
        if args["output"] is not None and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args["output"], fourcc, 30,
                (frame.shape[1], frame.shape[0]), True)

        
        # We refresh every 15 frames so that we can redetected the objects, then we track the detected objects
        # Hence we will be having 60 frame cycles
        if totalFrames%25 == 0:


            # Run detection algorithm
            with torch.no_grad():
                (num_objects,bboxes,scores,classes) = detect(img,rgb)


            # loop over the detections
            for i in np.arange(0, num_objects):
                # extract the confidence (i.e., probability) associated
                # with the prediction
                
                confidence = scores
                # Check for solid human detections
                # filter out weak detections by requiring
                # confidence
                if confidence[i] > args["confidence"] and classes[i] == 0:
                    # extract the index of the class label from the
                    # detections list
                    idx = int(classes[i])		
                    label = CLASSES[idx]

                    # if the class label is not a person, ignore it
                    if CLASSES[idx] != "person":
                        continue

                    # compute the (x, y)-coordinates of the bounding box
                    # for the object
                    box = bboxes[i]
                    (startX, startY, endX, endY) = np.array(box).astype("int")
                    bb = (startX, startY, endX, endY)
                    
                    
                    colour = fallAlgorithm(startX,startY,endX,endY)
                    checkSleep(bedbbox,bedCounter,bb)
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                        colour, 2)
                    cv2.putText(frame, label+ " " + str(5) , (startX, startY - 15),cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 2)
        cv2.rectangle(frame, (bedbbox[0][0], bedbbox[0][1]), (bedbbox[0][2], bedbbox[0][3]),(200,120,200), 2)
        cv2.putText(frame, str(bedCounter[0]), (bedbbox[0][1], bedbbox[0][1] - 5),
					cv2.FONT_HERSHEY_SIMPLEX, 0.20, (200,120,200), 1)    

        if writer is not None:
            writer.write(frame)
        
        # show the output frame
        cv2.imshow("Frame", frame)
        totalFrames += 1
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

        # update the FPS counter
        fps.update()

    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # check to see if we need to release the video writer pointer
    if writer is not None:
        writer.release()

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.release()
            


def driver():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", default = 0,
        help="path to input video file")
    ap.add_argument("-o", "--output", type=str,
        help="path to optional output video file")
    ap.add_argument("-c", "--confidence", type=float, default=0.01,
        help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())
    # Start streaming
    startStream(args)


if __name__ == '__main__':
    driver()