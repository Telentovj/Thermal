# USAGE
# py main.py --video video.mp4

# import the necessary packages
from datetime import datetime
from imutils.video import FPS
import argparse
import cv2
import logging
from helpers import *


def startStream(args):
    #Logger
    current = datetime.now().strftime('%Y-%m-%d-%H--%M--%S')
    logging.basicConfig(filename= "/home/vulcan/Documents/Thermal/Log/" + current + ".log", level=logging.INFO)

    try:
        #Track how many more frames to capture, if at the end of the required frames, we still have an 
        #anomally detection then we extend the frames required until we reach a point where frame 
        #required = 0 and no more anomally detections are found.
        anomallyFrameTracker = [0]
        accumulativeFrameQueue = []
        movingFrameQueue = []
        


        # initialize list of beds, each bed will be a series of bounding boxes.
        # xmin ymin xmax ymax
        bedbbox=[[30,45,80,65]]
        #bedbbox=[[10,80,35,118],[50,80,75,118],[90,80,115,118]]
        bedbboxID = ["D429F794-C889-4FC2-8439-00ABD9BC5F3C","24CB2FD9-B15C-49D7-9C8F-0DF7F3DB14D9","74922366-AA42-437F-9E6E-EFBEDF9700B6"]
        anomallyHistoryTracker = [0]*len(bedbbox)
        #0-255
        colourThreshold = args['colour']
        #Percentage of bed bbox needed to be covered by pixels with colour > 130 to be considered detection
        percentageThreshold = args['percentage']
        #Same as above but stricter for anomally detection
        percentageAnomallyThreshold = args['anomally']
        #Initialize the counter that will keep track of number of people on each bed.
        bedCounter = [0]*len(bedbbox)
        #Initialize the counter that will keep track of number of anomalous behaviour on each bed.
        bedAnomallyCounter = [0]*len(bedbbox)
        #Initialize the counter that will keep track of time spend on each bed.
        bedTime = [0]*len(bedbbox)
        
        # to track the frames
        totalFrames = 0


        # initialize the video stream and output video writer
        print("[INFO] starting video stream...")
        vs = cv2.VideoCapture(0)
        writer = None

        # start the frames per second throughput estimator
        fps = FPS().start()
        current = datetime.now().strftime('%Y-%m-%d-%H--%M--%S')
        logging.info(current + " Parameter Generation Success")
    except:
        current = datetime.now().strftime('%Y-%m-%d-%H--%M--%S')
        logging.info(current + " Parameter Generation Failed")


    # loop over frames from the video file stream
    while True:
        # grab the next frame from the video file
        (grabbed, frame) = vs.read()
        # check to see if we have reached the end of the video file
        if frame is None:
            current = datetime.now().strftime('%Y-%m-%d-%H--%M--%S')
            logging.info(current + " Video ended")
            break
        else:
            # Remove history, roughly ten minutes to remove completely
            for i in range(len(anomallyHistoryTracker)):
                if anomallyHistoryTracker[i] > 0:
                    anomallyHistoryTracker[i] -= 1

            # resize the frame for faster processing and then convert the
            frame = cv2.resize(frame, (128,128))
            colourFrame = frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bedbboxColours = getColours(frame,bedbbox)
            checkSleep(bedbboxColours,bedCounter,bedAnomallyCounter,colourThreshold,percentageThreshold,percentageAnomallyThreshold,bedTime)
            
            # frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            current = datetime.now().strftime('%Y-%m-%d-%H--%M--%S')
            logging.info(current + " Frame manipulations completed")
            # if we are supposed to be writing a video to disk, initialize
            # the writer
        if args["output"] is not None and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            writer = cv2.VideoWriter(args["output"], fourcc, 60,
                (frame.shape[1], frame.shape[0]), True)
            current = datetime.now().strftime('%Y-%m-%d-%H--%M--%S')
            logging.info(current + " Writer Initialized")

        for i in range(len(bedbbox)):
            if bedAnomallyCounter[i] > 0:
                cv2.rectangle(colourFrame, (bedbbox[i][0], bedbbox[i][1]), (bedbbox[i][2], bedbbox[i][3]),(0,0,255), 2)   
            else:
                cv2.rectangle(colourFrame, (bedbbox[i][0], bedbbox[i][1]), (bedbbox[i][2], bedbbox[i][3]),(255,255,255), 2)   
            if bedAnomallyCounter[i] > 0:
                cv2.putText(colourFrame,  str(round(bedTime[i],1)), (bedbbox[i][0], bedbbox[i][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.20, (0,0,255), 1) 
            else:
                cv2.putText(colourFrame,  str(round(bedTime[i],1)), (bedbbox[i][0], bedbbox[i][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.20, (255,255,255), 1) 
        current = datetime.now().strftime('%Y-%m-%d-%H--%M--%S')
        logging.info(current + " Bounding Box Drawn")

        if writer is not None:
            writer.write(colourFrame)
        
        # show the output frame
        cv2.imshow("Frame", cv2.resize(colourFrame,(640,640)))
        totalFrames += 1
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

        # update the FPS counter
        fps.update()

        # Call for frame requirements if there is an anomally
        if len(movingFrameQueue) > 50:
            movingFrameQueue.pop(0)
            current = datetime.now().strftime('%Y-%m-%d-%H--%M--%S')
            logging.info(current + " MovingFrameQueue Popped")
        else:
            movingFrameQueue.append(colourFrame)
            current = datetime.now().strftime('%Y-%m-%d-%H--%M--%S')
            logging.info(current + " MovingFrameQueue Appended")
        


        # Add frame requirements to track anomally
        bedAnomallyDetectedAt = [0] * len(bedbbox)
        for i in range(len(bedbbox)):
            if bedAnomallyCounter[i] > 0:
                bedAnomallyDetectedAt[i] = 1
                current = datetime.now().strftime('%Y-%m-%d-%H--%M--%S')
                logging.info(current + " Anomally Detected, starting call for more frames")
        
        # print(bedAnomallyDetectedAt)
        # print(anomallyHistoryTracker)
        
        for i in range(len(bedAnomallyDetectedAt)):
            if bedAnomallyDetectedAt[i] == 1:

                if anomallyFrameTracker[0] == 0 and anomallyHistoryTracker[i] == 0:
                    anomallyHistoryTracker[i] = args['timeSuppression'] #Roughly 10 minutes between tracks
                    for frame in movingFrameQueue:
                        accumulativeFrameQueue.append(frame)
                    anomallyFrameTracker[0] = 50
                
            current = datetime.now().strftime('%Y-%m-%d-%H--%M--%S')
            logging.info(current + " anomallyFrameTracker resetted to 200")

        # If we find a call for frame requirements, we append the frame then reduce the frame requirements incrementally
        if anomallyFrameTracker[0] != 0:
            accumulativeFrameQueue.append(colourFrame)
            anomallyFrameTracker[0] -= 1
        print(anomallyFrameTracker[0])
        try:
            if anomallyFrameTracker[0] == 0 and len(accumulativeFrameQueue) >= 51:
                anomallyfourcc = cv2.VideoWriter_fourcc(*'avc1')
                current = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                anomallyWriter = cv2.VideoWriter('/home/vulcan/Documents/Thermal/AnomallyVideo/' + current + ".mp4", anomallyfourcc , 10,
                    (128,128), True)
                for pic in  accumulativeFrameQueue:
                    anomallyWriter.write(pic)
                anomallyWriter.release()
                accumulativeFrameQueue = []
                print(current)
                updateAnomalousBehaviour(current)
        except:
            current = datetime.now().strftime('%Y-%m-%d-%H--%M--%S')
            logging.info(current + " Anomally Video Output unsuccessful")


    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    current = datetime.now().strftime('%Y-%m-%d-%H--%M--%S')
    logging.info(current + "[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    logging.info(current + "[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # check to see if we need to release the video writer pointer
    if writer is not None:
        writer.release()
    if len(accumulativeFrameQueue) > 0 and anomallyFrameTracker[0] != 0:
        try:
            anomallyfourcc = cv2.VideoWriter_fourcc(*'avc1')
            current = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            anomallyWriter = cv2.VideoWriter( '/home/vulcan/Documents/Thermal/AnomallyVideo/' + current + ".mp4", anomallyfourcc , 10,
                (128,128), True)
            for pic in  accumulativeFrameQueue:
                anomallyWriter.write(pic)
            anomallyWriter.release()
            accumulativeFrameQueue = []
            updateAnomalousBehaviour(current)
        except:
            current = datetime.now().strftime('%Y-%m-%d-%H--%M--%S')
            logging.info(current + " Anomally Video Output unsuccessful")
    
    updateBedTime(bedTime,bedbboxID)
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
    ap.add_argument("-c", "--colour", default = 90, 
        help="Value from 0-255 to represent colour")
    ap.add_argument("-n", "--percentage", default = 0.35, 
        help="percentage of bed bb required to consider normal detection")
    ap.add_argument("-a", "--anomally", default = 0.50,
        help="percentage of bed bb required to consider anomalous detection")
    ap.add_argument("-t", "--timeSuppression", default = 31860,
        help="Time between anomalous detections for single bed, default at 1 hr, 31860 frames")
    args = vars(ap.parse_args())
    # Start streaming
    startStream(args)


if __name__ == '__main__':
    driver()
