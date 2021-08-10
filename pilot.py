#!/usr/bin/env python3
# USAGE
# py main.py --video video.mp4

# import the necessary packages
from datetime import datetime
from imutils.video import FPS
import argparse
import cv2
import logging
import time
import numpy as np
import os
import json
from imutils import video
import requests



# add parameter for timeout
# change time sent to server into minutes.

class ThermalSensor:
    
    # This is used to track remaining the number of frames required to be captured
    # This is defaulted to 50.
    # The index represent the bed position
    anomallyFrameTracker = [0]

    # This is used to compile all the frames of the incident
    accumulativeFrameQueue = []

    # This is used as a queue for the past 50 frames from the current frame. 
    # This is added to the accumulativeFrameQueue so that we can see what happens 
    # prior to anomally detection
    movingFrameQueue = []

    # This represent the hardcoded bed bounding boxes
    bedbbox=[[35,70,70,105]]

    # This represent the ID associated to each bed.
    # Important for posting the data to SmartCaire
    bedbboxID = ["D429F794-C889-4FC2-8439-00ABD9BC5F3C","24CB2FD9-B15C-49D7-9C8F-0DF7F3DB14D9","74922366-AA42-437F-9E6E-EFBEDF9700B6"]
        
    # This is to track time since the last anomally that have happened at each bed
    # Ensures that incidents sent are at least 10 minutes apart
    anomallyHistoryTracker = [0]*len(bedbbox)

    # Initialize the counter that will keep track of number of people on each bed.
    bedCounter = [0]*len(bedbbox)
    
    # Initialize the counter that will keep track of number of anomalous behaviour on each bed.
    bedAnomallyCounter = [0]*len(bedbbox)
    
    # Initialize the counter that will keep track of time spend on each bed.
    bedTime = [0]*len(bedbbox)
    
    # To track the frames
    totalFrames = 0
    
    # Counter to track consecutive detections
    consecutiveDetectionCount = 0
    
    # Counter to track consecutive anomalous detections
    consecutiveAnomallyDetectionCount = 0

    def __init__(self,colour,percentage,anomally,minTimeDetect,output,timeSuppression, halfVideo, timeOut):
        #Logger
        current = datetime.now().strftime('%Y-%m-%d-%H--%M--%S')
        logging.basicConfig(filename= "/home/vulcan/Documents/Thermal/Log/" + current + ".log", level=logging.INFO)
        
        #output file name
        self.output = output

        #Duration for half of the video, used to track the frames
        self.halfVideo = halfVideo

        # Time suppression
        self.timeSuppression = timeSuppression

        # Presents threshold (0-1) for normalized colour
        self.colourThreshold = colour
        
        # Percentage of bed bbox needed to be covered by pixels with normalized colour > colourThreshold to be considered detection
        self.percentageThreshold = percentage
        
        # Same as above but stricter for anomally detection
        self.percentageAnomallyThreshold = anomally
        
        # Minimum time required for something to be considered a detection
        self.minTime = minTimeDetect

        # Time program should be ran
        self.timeOut = timeOut

        # initialize the video stream and output video writer
        print("[INFO] starting video stream...")
        self.vs = cv2.VideoCapture(0)
        self.writer = None

        # start the frames per second throughput estimator
        self.fps = FPS().start()
        self.current = datetime.now().strftime('%Y-%m-%d-%H--%M--%S')
        logging.info(current + " Parameter Generation Success")
    
    def getColours(self,frame):
        bedbbox = self.bedbbox
        try:
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
            current = datetime.now().strftime('%Y-%m-%d-%H--%M--%S')
            logging.info(current + " Colour Conversion Completed")
            return buffer
        except:
            current = datetime.now().strftime('%Y-%m-%d-%H--%M--%S')
            logging.error(current + " Colour Conversion Failed")

    def getNormalizedColours(self,colors):
        buffer = []
        for i in range(len(colors)):
            buffer.append([])
        # Loop for each bed bounding box
        for i in range(len(colors)):
            #Get maximum value & minimum value
            maxValue = 0
            minValue = 255
            for j in range(len(colors[i])):
                if colors[i][j] > maxValue:
                    maxValue = colors[i][j]
                if colors[i][j] < minValue:
                    minValue = colors[i][j]
            #Get Normalized Values
            #Add it into new normalizd number list
            for k in range(len(colors[i])):
                #Solve edge case where maxValue == minValue resulting in division via zero.
                if maxValue != minValue:
                    normalNumber = (colors[i][k]-minValue)/(maxValue-minValue)
                    buffer[i].append(normalNumber)
                else:
                    buffer[i].append(0)
        return buffer

    def getDifferenceFromMean(self,colors):
        # Colors is a list of list where there is a list for each bed and each bed has a list of numbers indicating colour within bedbbox
        buffer = []
        for i in range(len(colors)):
            totalValue = 0
            buffer.append([])
            for j in range(len(colors[i])):
                totalValue += colors[i][j]
            meanValue = totalValue/len(colors[i])
            for j in range(len(colors[i])):
                buffer[i].append(colors[i][j]-meanValue)
        return buffer

    def checkSleep(self,bedbboxColours):
        bedCounter = self.bedCounter
        bedAnomallyCounter = self.bedAnomallyCounter
        colourThreshold = self.colourThreshold
        percentageThreshold = self.percentageThreshold
        percentageAnomallyThreshold = self.percentageAnomallyThreshold
        bedTime = self.bedTime
        consecutiveDetectionCount = self.consecutiveDetectionCount
        consecutiveAnomallyDetectionCount = self.consecutiveAnomallyDetectionCount
        minTime = self.minTime
        try:
            for i in range(len(bedbboxColours)):
                detectionCount = 0
                countNeeded = percentageThreshold*len(bedbboxColours[i])
                anomallyCountNeeded = percentageAnomallyThreshold*len(bedbboxColours[i])
                for j in range(len(bedbboxColours[i])):
                    if bedbboxColours[i][j] > colourThreshold:
                        detectionCount += 1
                print(detectionCount)
                print(countNeeded)
                if(detectionCount > countNeeded):
                    bedCounter[i] = 1
                    consecutiveDetectionCount += 1
                    if consecutiveDetectionCount > minTime:
                        if consecutiveDetectionCount == (minTime+1):
                            bedTime[i] += minTime/8.8
                        bedTime[i] += 1/8.8
                    if(detectionCount > anomallyCountNeeded):
                        consecutiveAnomallyDetectionCount += 1
                        if consecutiveAnomallyDetectionCount > minTime:
                            bedAnomallyCounter[i] = 1
                    else:
                        consecutiveAnomallyDetectionCount = 0 
                        bedAnomallyCounter[i] = 0
                else:
                    consecutiveDetectionCount = 0
                    bedAnomallyCounter[i] = 0
                    bedCounter[i] = 0
            current = datetime.now().strftime('%Y-%m-%d-%H--%M--%S')
            logging.info(current + " Sleep Check Completed")
            return consecutiveDetectionCount,consecutiveAnomallyDetectionCount
        except:
            current = datetime.now().strftime('%Y-%m-%d-%H--%M--%S')
            logging.error(current + " Sleep Check Failed")

    def updateAnomalousBehaviour(self,videoName):
        try:
            url = "https://smartcaire-dev.azurewebsites.net/api/v2.0/auth/login"
            payload = json.dumps({
                "email": "sarthak.ganoorkar@vulcan-ai.com",
                "password": "VisionAI@052020"
                })
            headers = {'Content-Type': 'application/json'}

            response = requests.request("POST", url, headers=headers, data=payload)
            accessToken = response.json().get('accessToken')
            current = datetime.now().strftime('%Y-%m-%d-%H--%M--%S')
            logging.info(current + " Getting Auth Token Success")
        except:
            current = datetime.now().strftime('%Y-%m-%d-%H--%M--%S')
            logging.error(current + " Getting Auth Token Failure")


        try:
            #Upload anomalous behaviour detection
            #status_id: 1-Aggression, 2-Fall, 3-Anomalous Behaviour, 4-Erratic Behaviour
            #Detected by device id: 2-Camera, 3-Sensor

            url = "https://smartcaire-dev.azurewebsites.net/api/v2.0/incident"
            payload={'incident_type_id': '3',
                    'incident_dt': videoName, 
                    'status_id': '1',
                    'detected_by_device_id': '3', #2 is camera, 3 is sensor
                    'sensor_id': '3', #based on sensor in the dashboard
                    'media_type': 'video',
                    'location': 'Dormitory'}
            files=[
                    ('file',(videoName + '.mp4',open('/home/vulcan/Documents/Thermal/AnomallyVideo/' + videoName + '.mp4','rb'),'application/octet-stream'))
                    ]
            headers = {
                    'Authorization': 'Bearer {}'.format(accessToken),
                    }

            response = requests.request("POST", url, headers=headers, data=payload, files=files)

            print(response.text)
            current = datetime.now().strftime('%Y-%m-%d-%H--%M--%S')
            logging.info(current + " Anomalous Detection Video Upload Completed")
        except:
            current = datetime.now().strftime('%Y-%m-%d-%H--%M--%S')
            logging.error(current + " Anomalous Detection Video Upload Failed")

    def updateBedTime(self):
        try:
            bedTime = self.bedTime/60
            bedbboxID = self.bedbboxID
            url = "https://smartcaire-dev.azurewebsites.net/api/v2.0/auth/login"
            payload = json.dumps({
                "email": "sarthak.ganoorkar@vulcan-ai.com",
                "password": "VisionAI@052020"
                })
            headers = {'Content-Type': 'application/json'}

            response = requests.request("POST", url, headers=headers, data=payload)
            accessToken = response.json().get('accessToken')
            current = datetime.now().strftime('%Y-%m-%d-%H--%M--%S')
            logging.info(current + " Getting Auth Token Success")
        except:
            current = datetime.now().strftime('%Y-%m-%d-%H--%M--%S')
            logging.error(current + " Getting Auth Token Failure")

        try:
            url = "https://smartcaire-dev.azurewebsites.net/api/v2.0/user/time-in-bed"
            payload={"user_id": "D429F794-C889-4FC2-8439-00ABD9BC5F3C"}
            headers = {
            'Authorization': 'Bearer {}'.format(accessToken)
            }
            response = requests.request("GET", url, headers=headers, data=payload)
            print(response.text)

            for i in range(len(bedTime)):
                current = datetime.now().strftime('%Y-%m-%d')
                # Save the time spend in bed
                url = "https://smartcaire-dev.azurewebsites.net/api/v2.0/user/time-in-bed"
                payload = json.dumps({
                    "user_id": bedbboxID[i],
                    "datetime": current,
                    "time_spent": bedTime[i]  # hours/minutes?
                    })

                headers = {'Authorization': 'Bearer {}'.format(accessToken),
                        'Content-Type': 'application/json',
                        }
                response = requests.request("POST", url, headers=headers, data=payload)
                print(response.text)
                current = datetime.now().strftime('%Y-%m-%d-%H--%M--%S')
                logging.info(current + " BedTime Upload Completed")
        except:
            current = datetime.now().strftime('%Y-%m-%d-%H--%M--%S')
            logging.error(current + " BedTime Upload Failed")

    def startStream(self):
        timeout = 60*60*self.timeOut
        timeout_start = time.time()
        # loop over frames from the video file stream
        while time.time() < timeout_start+timeout:
            # grab the next frame from the video file
            (grabbed, frame) = self.vs.read()
            # check to see if we have reached the end of the video file
            if frame is None:
                current = datetime.now().strftime('%Y-%m-%d-%H--%M--%S')
                logging.info(current + " Video ended")
                break
            else:
                # Remove history, roughly ten minutes to remove completely
                for i in range(len(self.anomallyHistoryTracker)):
                    if self.anomallyHistoryTracker[i] > 0:
                        self.anomallyHistoryTracker[i] -= 1

                # resize the frame for faster processing and then convert the
                frame = cv2.resize(frame, (128,128))
                colourFrame = frame
                
                # For streaming on website
                streaming = colourFrame
                cv2.putText(streaming, datetime.now().strftime('%Y-%m-%d-%H--%M--%S'), (15,25), cv2.FONT_HERSHEY_SIMPLEX, 0.20, (255,255,255), 1) 
                cv2.imwrite('/home/vulcan/Documents/Thermal/website/streaming-server-thk/image_dump/thermalJPG.jpeg',colourFrame)

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Check sleep and anomally here
                # Get colours
                bedbboxColours = self.getColours(frame)
                # Get Normalized Colours
                # bedbboxColours = self.getNormalizedColours(bedbboxColours)
                # Get difference from mean colors
                bedbboxColours = self.getDifferenceFromMean(bedbboxColours)
                # Get Detections and Anomally Detections
                consecutiveDetectionCount,consecutiveAnomallyDetectionCount = self.checkSleep(bedbboxColours)
                
                # frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                current = datetime.now().strftime('%Y-%m-%d-%H--%M--%S')
                logging.info(current + " Frame manipulations completed")
                # if we are supposed to be writing a video to disk, initialize
                # the writer
            if self.output is not None and self.writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
                self.writer = cv2.VideoWriter(args["output"], fourcc, 60,
                    (frame.shape[1], frame.shape[0]), True)
                current = datetime.now().strftime('%Y-%m-%d-%H--%M--%S')
                logging.info(current + " Writer Initialized")

            for i in range(len(self.bedbbox)):
                if self.bedAnomallyCounter[i] > 0:
                    cv2.rectangle(colourFrame, (self.bedbbox[i][0], self.bedbbox[i][1]), (self.bedbbox[i][2], self.bedbbox[i][3]),(0,0,255), 2)   
                else:
                    cv2.rectangle(colourFrame, (self.bedbbox[i][0], self.bedbbox[i][1]), (self.bedbbox[i][2], self.bedbbox[i][3]),(255,255,255), 2)   
                if self.bedAnomallyCounter[i] > 0:
                    cv2.putText(colourFrame,  str(round(self.bedTime[i],1)), (int(self.bedbbox[i][0]+(self.bedbbox[i][2]-self.bedbbox[i][0])/2), int(self.bedbbox[i][1]+(self.bedbbox[i][3]-self.bedbbox[i][1])/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.20, (0,0,255), 1) 
                else:
                    cv2.putText(colourFrame,  str(round(self.bedTime[i],1)), (int(self.bedbbox[i][0]+(self.bedbbox[i][2]-self.bedbbox[i][0])/2), int(self.bedbbox[i][1]+(self.bedbbox[i][3]-self.bedbbox[i][1])/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.20, (0,255,60), 1) 
            current = datetime.now().strftime('%Y-%m-%d-%H--%M--%S')
            logging.info(current + " Bounding Box Drawn")

            if self.writer is not None:
                self.writer.write(colourFrame)
            
            # show the output frame
            cv2.imshow("Frame", cv2.resize(colourFrame,(640,640)))
          
            self.totalFrames += 1
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

            # update the FPS counter
            self.fps.update()

            # Call for frame requirements if there is an anomally
            if len(self.movingFrameQueue) > 50:
                self.movingFrameQueue.pop(0)
                current = datetime.now().strftime('%Y-%m-%d-%H--%M--%S')
                logging.info(current + " MovingFrameQueue Popped")
            else:
                self.movingFrameQueue.append(colourFrame)
                current = datetime.now().strftime('%Y-%m-%d-%H--%M--%S')
                logging.info(current + " MovingFrameQueue Appended")
            


            # Add frame requirements to track anomally
            bedAnomallyDetectedAt = [0] * len(self.bedbbox)
            for i in range(len(self.bedbbox)):
                if self.bedAnomallyCounter[i] > 0:
                    bedAnomallyDetectedAt[i] = 1
                    current = datetime.now().strftime('%Y-%m-%d-%H--%M--%S')
                    logging.info(current + " Anomally Detected, starting call for more frames")
            

            
            for i in range(len(bedAnomallyDetectedAt)):
                if bedAnomallyDetectedAt[i] == 1:

                    if self.anomallyFrameTracker[0] == 0 and self.anomallyHistoryTracker[i] == 0:
                        self.anomallyHistoryTracker[i] = self.timeSuppression #Roughly 10 minutes between tracks
                        for frame in self.movingFrameQueue:
                            self.accumulativeFrameQueue.append(frame)
                        self.anomallyFrameTracker[0] = self.halfVideo
                        current = datetime.now().strftime('%Y-%m-%d-%H--%M--%S')
                        logging.info(current + " anomallyFrameTracker resetted to 50")

            # If we find a call for frame requirements, we append the frame then reduce the frame requirements incrementally
            if self.anomallyFrameTracker[0] != 0:
                self.accumulativeFrameQueue.append(colourFrame)
                self.anomallyFrameTracker[0] -= 1

            try:
                if self.anomallyFrameTracker[0] == 0 and len(self.accumulativeFrameQueue) >= 51:
                    anomallyfourcc = cv2.VideoWriter_fourcc(*'avc1')
                    current = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    anomallyWriter = cv2.VideoWriter('/home/vulcan/Documents/Thermal/AnomallyVideo/' + current + ".mp4", anomallyfourcc , 10,
                        (128,128), True)
                    for pic in self.accumulativeFrameQueue:
                        anomallyWriter.write(pic)
                    anomallyWriter.release()
                    self.accumulativeFrameQueue = []
                    self.updateAnomalousBehaviour(current)
            except:
                current = datetime.now().strftime('%Y-%m-%d-%H--%M--%S')
                logging.info(current + " Anomally Video Output unsuccessful")
            # stop the timer and display FPS information

        self.fps.stop()
        print("[INFO] elapsed time: {:.2f}".format(self.fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(self.fps.fps()))
        current = datetime.now().strftime('%Y-%m-%d-%H--%M--%S')
        logging.info(current + "[INFO] elapsed time: {:.2f}".format(self.fps.elapsed()))
        logging.info(current + "[INFO] approx. FPS: {:.2f}".format(self.fps.fps()))

        # check to see if we need to release the video writer pointer
        if self.writer is not None:
            self.writer.release()
        if len(self.accumulativeFrameQueue) > 0 and self.anomallyFrameTracker[0] != 0:
            try:
                anomallyfourcc = cv2.VideoWriter_fourcc(*'avc1')
                current = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                anomallyWriter = cv2.VideoWriter( '/home/vulcan/Documents/Thermal/AnomallyVideo/' + current + ".mp4", anomallyfourcc , 10,
                    (128,128), True)
                for pic in self.accumulativeFrameQueue:
                    anomallyWriter.write(pic)
                anomallyWriter.release()
                self.accumulativeFrameQueue = []
                self.updateAnomalousBehaviour(current)
            except:
                current = datetime.now().strftime('%Y-%m-%d-%H--%M--%S')
                logging.info(current + " Anomally Video Output unsuccessful")

        self.updateBedTime()
        # do a bit of cleanup
        cv2.destroyAllWindows()
        self.vs.release()

def driver():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", default = 0,
        help="path to input video file")
    ap.add_argument("-o", "--output", type=str,
        help="path to optional output video file")
    ap.add_argument("-c", "--colour", default = 0, 
        help="Value to serve as the cut off for when we consider a pixel a human")
    ap.add_argument("-n", "--percentage", default = 0.28, 
        help="percentage of bed bb required to consider normal detection")
    ap.add_argument("-a", "--anomally", default = 0.5,
        help="percentage of bed bb required to consider anomalous detection")
    ap.add_argument("-t", "--timeSuppression", default = 31860,
        help="Time between anomalous detections for single bed, default at 1 hr, 31860 frames")
    ap.add_argument("-tt", "--minTimeDetect", default = 0,
        help="Minimum time required for something to be considered a detection"),
    ap.add_argument("-hv", "--halfVideo", default = 50,
        help="number of frames to collect for the first half of anomally video"),
    ap.add_argument("-to", "--timeOut", default = 10,
        help="hours the program should be ran")

    args = vars(ap.parse_args())
    # Start streaming
    
    ts = ThermalSensor(args['colour'],args['percentage'],args['anomally'],args['minTimeDetect'],args['output'],args['timeSuppression'],args['halfVideo'],args['timeOut'])
    ts.startStream()


if __name__ == '__main__':
    driver()