# USAGE
# py main.py --video video.avi

# import the necessary packages
from imutils.video import FPS
import argparse
import cv2
from helpers import *

def startStream(args):

    frameQueue = []
    # initialize list of beds, each bed will be a series of bounding boxes.
    # xmin ymin xmax ymax
    bedbbox=[[50,90,75,118]]
    # bedbbox=[[10,80,35,118],[50,80,75,118],[90,80,115,118]]
    #0-255
    colourThreshold = 130
    percentageThreshold = 0.3
    percentageAnomallyThreshold = 0.5
    bedCounter = [0]*len(bedbbox)
    bedAnomallyCounter = [0]*len(bedbbox)
    bedTime = [0]*len(bedbbox)
    
    # to track the frames
    totalFrames = 0


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
            bedbboxColours = getColours(frame,bedbbox)
            checkSleep(bedbboxColours,bedCounter,bedAnomallyCounter,colourThreshold,percentageThreshold,percentageAnomallyThreshold,bedTime)
            
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
 
            # if we are supposed to be writing a video to disk, initialize
            # the writer
        if args["output"] is not None and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args["output"], fourcc, 10,
                (frame.shape[1], frame.shape[0]), True)


        for i in range(len(bedbbox)):
            if bedAnomallyCounter[i] > 0:
                cv2.rectangle(frame, (bedbbox[i][0], bedbbox[i][1]), (bedbbox[i][2], bedbbox[i][3]),(0,0,255), 2)   
            else:
                cv2.rectangle(frame, (bedbbox[i][0], bedbbox[i][1]), (bedbbox[i][2], bedbbox[i][3]),(200,120,200), 2)   
            if bedAnomallyCounter[i] > 0:
                cv2.putText(frame,  str(round(bedTime[i],1)), (bedbbox[i][0], bedbbox[i][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.20, (0,0,255), 1) 
            else:
                cv2.putText(frame,  str(round(bedTime[i],1)), (bedbbox[i][0], bedbbox[i][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.20, (200,120,200), 1) 
               

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

        if len(frameQueue) > 200:
            frameQueue.pop()
        else:
            frameQueue.append(frame)
        
        if bedAnomallyCounter[i] > 0:
            print('Save Video')
            #updateAnomalousBehaviour(video)
        #Save video code

    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # check to see if we need to release the video writer pointer
    if writer is not None:
        writer.release()
    # updateBedTime(bedTime)
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
