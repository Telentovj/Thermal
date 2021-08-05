import json
from imutils import video
import requests
import logging
from datetime import datetime
def getColours(frame):
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

def getNormalizedColours(colors):
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
            normalNumber = (colors[i][k]-minValue)/(maxValue-minValue)
            buffer[i].append(normalNumber)
        print(maxValue)
        print(minValue)
    return buffer




def checkSleep(bedbboxColours):
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



def updateAnomalousBehaviour(videoName):
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


def updateBedTime():
    try:
        bedTime = self.bedTime
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

        logging.error(current + " Getting Auth Token Failure")

    try:
        # url = "https://smartcaire-dev.azurewebsites.net/api/v2.0/user/time-in-bed"
        # payload={"user_id": "D429F794-C889-4FC2-8439-00ABD9BC5F3C"}
        # headers = {
        # 'Authorization': 'Bearer {}'.format(accessToken)
        # }
        # response = requests.request("GET", url, headers=headers, data=payload)
        # print(response.text)

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

