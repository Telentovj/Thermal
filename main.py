# USAGE
# py main.py --video video.avi

# import the necessary packages
from imutils.video import FPS
import argparse
import cv2

    



def getColours(frame, bedbbox):
    buffer = []
    for i in range(len(bedbbox)):
        buffer.append([])
    
    # j is x , k is y
    # frmae[y][x]
    for i in range(len(bedbbox)):
        count = 0
        for j in range(bedbbox[i][0], bedbbox[i][2]):
            for k in range(bedbbox[i][1], bedbbox[i][3]):
                count+=1
                buffer[i].append(frame[k][j])
        
    return buffer





def checkSleep(bedbboxColours,bedCounter,bedAnomallyCounter,colourThreshold,percentageThreshold,percentageAnomallyThreshold,bedTime):
    
    for i in range(len(bedbboxColours)):
        detectionCount = 0
        countNeeded = percentageThreshold*len(bedbboxColours[i])
        anomallyCountNeeded = percentageAnomallyThreshold*len(bedbboxColours[i])
        for j in range(len(bedbboxColours[i])):
            if bedbboxColours[i][j] > colourThreshold:
                detectionCount += 1
        if(detectionCount > countNeeded):
            bedCounter[i] = 1
            bedTime[i] += 0.1136
            if(detectionCount > anomallyCountNeeded):
                bedAnomallyCounter[i] = 1
            else:
                bedAnomallyCounter[i] = 0
        else:
            bedAnomallyCounter[i] = 0
            bedCounter[i] = 0
        
import requests

def updateApi(bedTime):
            url = "http://work-smart-dev.azurewebsites.net/api/v2.0/auth/login"
            payload="{\n    \"email\" : \"camera-vulcan@vulcan-ai.com\",\n    \"password\":\"VisionAI@052020\"\n}"
            headers = {'Content-Type': 'application/json'}

            response = requests.request("POST", url, headers=headers, data=payload)
            accessToken = response.json().get('accessToken')

            url = "http://work-smart-dev.azurewebsites.net/api/v2.0/incident"
            payload = {}
            files = [bedTime]
            headers = {'Authorization': 'Bearer {}'.format(accessToken)}
            response = requests.request("POST", url, headers=headers, data=payload, files=files)
        
        
 
        
def startStream(args):

    # initialize list of beds, each bed will be a series of bounding boxes.
    # xmin ymin xmax ymax
    # 2 & 3 shift left by 5 pxiels
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
 

            
            # print(frame.shape)
            # print(np.transpose(image_data[0]).shape)
            # if we are supposed to be writing a video to disk, initialize
            # the writer
        if args["output"] is not None and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args["output"], fourcc, 10,
                (frame.shape[1], frame.shape[0]), True)

        
        # We refresh every 15 frames so that we can redetected the objects, then we track the detected objects
        # Hence we will be having 60 frame cycles
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

    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # check to see if we need to release the video writer pointer
    if writer is not None:
        writer.release()
    # updateApi(bedTime)
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