import numpy as np
import cv2 as cv
import handDeModule as htm
import time
import autopy 

wCam,hCam = 640,480

cap = cv.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)
pTime = 0
detector = htm.handDetector(maxHands=1)
while True:
    #1. Find Hand Landmarks
    success,img = cap.read()
    img = detector.findHands()
    lmList,bbox = detector.find
    
    #2.get tip of the fingers
    #3.which fingers are up
    #4.mode (moving or clicking)
    #5.converting coordinates
    #6.Smothen Values
    #7.Move Mouse
    #8.check if clicking
    #9.disttance between the mouse
    #10.if less distance click
    
    #11.Frame Rate
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv.putText(img,f'FPS:{int(fps)}',(30,50),cv.FONT_HERSHEY_COMPLEX,1,
               (0,255,0),2)
    
    #12.Display
    cv.imshow("img",img)
    cv.waitKey(1)
    
    