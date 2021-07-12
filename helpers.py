import json
import requests
import logging
from datetime import datetime
def getColours(frame, bedbbox):
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

def checkSleep(bedbboxColours,bedCounter,bedAnomallyCounter,colourThreshold,percentageThreshold,percentageAnomallyThreshold,bedTime):
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
                bedTime[i] += 1/8.8
                if(detectionCount > anomallyCountNeeded):
                    bedAnomallyCounter[i] = 1
                else:
                    bedAnomallyCounter[i] = 0
            else:
                bedAnomallyCounter[i] = 0
                bedCounter[i] = 0
        current = datetime.now().strftime('%Y-%m-%d-%H--%M--%S')
        logging.info(current + " Sleep Check Completed")
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
        #Upload fall detection
        #status_id: 1-Aggression, 2-Fall, 3-Anomalous Behaviour, 4-Erratic Behaviour
        #Detected by device id: 2-Camera, 3-Sensor
        
        url = "https://smartcaire-dev.azurewebsites.net/api/v2.0/incident"
        payload={'incident_type_id': '2',
        'incident_dt': '2021-07-09 10:25:14.457',
        'status_id': '1',
        'detected_by_device_id': '3', #2 is camera, 3 is sensor
        'sensor_id': '3', #based on sensor in the dashboard
        'media_type': 'video',
        'location': 'Workplace'}
        files=[
            ('file',(videoName + '.avi',open('/home/vulcan/Documents/Thermal/AnomallyVideo/' + videoName + '.avi','rb'),'application/octet-stream'))
        ]
        headers = {
            'Authorization': 'Bearer {}'.format(accessToken),
            'Content-Type': 'application/json',
        }

        response = requests.request("POST", url, headers=headers, data=payload, files=files)

        print(response.text)
        current = datetime.now().strftime('%Y-%m-%d-%H--%M--%S')
        logging.info(current + " Anomalous Detection Video Upload Completed")
    except:
        current = datetime.now().strftime('%Y-%m-%d-%H--%M--%S')
        logging.error(current + " Anomalous Detection Video Upload Failed")


def updateBedTime(bedTime):
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
        # Save the time spend in bed
        url = "https://smartcaire-dev.azurewebsites.net/api/v2.0/user/time-in-bed"
        payload = json.dumps({
            "user_id": "d429f794-c889-4fc2-8439-00abd9bc5f3c",
            "datetime": "2021-07-09",
            "time_spent": bedTime  # hours/minutes?
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

