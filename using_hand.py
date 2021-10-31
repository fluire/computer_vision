import hand_trrackingmin as htm
import cv2 as cv
import time

ptime = 0
ctime = 0
obj = cv.VideoCapture(0)
detector = htm.handdetector()
while True:
    success,img = obj.read()
    img = detector.findHands(img)   
    lmList = detector.handPos(img,8)
    
    if len(lmList)!=0:
        print(lmList[4])
    
    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime
    
    cv.putText(img,str(int(fps)),(10,70),cv.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)
    cv.rectangle(img,(50,50),(300,300),(40,200,127),2)
    cv.imshow("image",img)
    cv.waitKey(1)
    