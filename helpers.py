import json
import requests

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
        

def updateAnomalousBehaviour(video):
    url = "https://smartcaire-dev.azurewebsites.net/api/v2.0/auth/login"
    payload = json.dumps({
        "email": "sarthak.ganoorkar@vulcan-ai.com",
        "password": "VisionAI@052020"
    })
    headers = {'Content-Type': 'application/json'}

    response = requests.request("POST", url, headers=headers, data=payload)
    accessToken = response.json().get('accessToken')
    print(accessToken)

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
        ('file',('snippet.mp4',open('/C:/Users/sarth/Vulcan/SmartCaire/Docs/snippet.mp4','rb'),'application/octet-stream'))
    ]
    headers = {
        'Authorization': 'Bearer {}'.format(accessToken),
        'Content-Type': 'application/json',
    }
    headers = {
        'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6InNhcnRoYWsuZ2Fub29ya2FyQHZ1bGNhbi1haS5jb20iLCJ1c2VyX2lkIjoiRDQyOUY3OTQtQzg4OS00RkMyLTg0MzktMDBBQkQ5QkM1RjNDIiwiY2xpZW50X2lkIjoiRTQzMzNDREItMDgzRi00NDYyLTgxNkItNjIwMTkxMDNEMTg5Iiwicm9sZSI6IkFkbWluIiwiZmlyc3RfbmFtZSI6IlNhcnRoYWsiLCJsYXN0X25hbWUiOiJHYW5vb3JrYXIiLCJyb2xlX2lkIjozLCJtYWNfaWQiOm51bGwsIndlYXJhYmxlX2lkIjpudWxsLCJ3ZWFyYWJsZV9uYW1lIjpudWxsLCJ3ZWFyYWJsZV9tb2RlbF9pZCI6bnVsbCwid2VhcmFibGVfbW9kZWxfbmFtZSI6bnVsbCwiaWF0IjoxNjI1Nzk3MTIxLCJleHAiOjE2MjU4ODM1MjF9.iptgJFHfkeUnEMZ4mfg3kB2mkDlRhO-XKKZcMTXQh5o'
    }

    response = requests.request("POST", url, headers=headers, data=payload, files=files)

    print(response.text)

def updateBedTime(bedTime):
    url = "https://smartcaire-dev.azurewebsites.net/api/v2.0/auth/login"
    payload = json.dumps({
        "email": "sarthak.ganoorkar@vulcan-ai.com",
        "password": "VisionAI@052020"
    })
    headers = {'Content-Type': 'application/json'}

    response = requests.request("POST", url, headers=headers, data=payload)
    accessToken = response.json().get('accessToken')
    print(accessToken)

    # Save the time spend in bed
    url = "https://smartcaire-dev.azurewebsites.net/api/v2.0/user/time-in-bed"
    payload = json.dumps({
        "user_id": "d429f794-c889-4fc2-8439-00abd9bc5f3c",
        "datetime": "2021-07-09",
        "time_spent": 6656  # hours/minutes?
    })
        
    headers = {'Authorization': 'Bearer {}'.format(accessToken),
                'Content-Type': 'application/json',
            }
    response = requests.request("POST", url, headers=headers, data=payload)

    # Save video of anomalous behaviour