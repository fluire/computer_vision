import cv2 as cv
import time
import numpy as np
import handDeModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

#####################################
wCam,hCam = 1280,720
#####################################

cap = cv.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)
cTime = 0
pTime = 0


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
#volume.GetMute()
#volume.GetMasterVolumeLevel()
vol = volume.GetVolumeRange()
volume.SetMasterVolumeLevel(-5, None)
minvol = vol[0]
maxvol = vol[1]

detector = htm.handDetector(detectionCon=0.7)

while True:
    success,img = cap.read()
    img = detector.findHands(img)
    lmList = detector.handPos(img,draw=False)
    vol=0
    volBar = 400
    if len(lmList)>0:
        
        
        x1 , y1 = lmList[4][1],lmList[4][2]
        x2 , y2 = lmList[8][1],lmList[8][2]
        cx,cy = (x1+x2)//2,(y1+y2)//2
        
        cv.circle(img,(x1,y1),15,(255,0,255),cv.FILLED)
        cv.circle(img,(x2,y2),15,(255,0,255),cv.FILLED)
        cv.circle(img,(cx,cy),15,(255,0,255),cv.FILLED)
        cv.line(img,(x1,y1),(x2,y2),(255,100,150),thickness = 3,lineType=1)
        
        length = math.hypot(x2-x1,y2-y1)
          
        #hand range = 50-300
        #vol range = -65 - 0
        
        
        vol = np.interp(length,[50,360],[minvol,maxvol])
        volBar = np.interp(length,[50,350],[400,150])
        
        volume.SetMasterVolumeLevel(vol,None)

    cv.rectangle(img,(50,150),(85,400),(0,200,50),3)   
    cv.rectangle(img,(50,int(volBar)),(85,400),(0,200,50),cv.FILLED)   

    cTime = time.time()
    fps  = 1/(cTime-pTime)
    pTime = time.time()
    
    cv.putText(img,f'FPS: {int(fps)}',(40,70),cv.FONT_HERSHEY_COMPLEX
               ,1,(100,255,0),3)
    cv.imshow("img",img)
    cv.waitKey(1)