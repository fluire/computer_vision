import cv2 as cv
import  mediapipe as mp
import time

class poseDetector():
    
    def __init__(self,mode= False,upper_body =  False,smooth = True , detectionCon = 0.5,trackCon=0.5):
        self.mode         = mode
        self.upper_body   =upper_body
        self.smooth       = smooth
        self.upper_body   = False
        self.detectionCon = detectionCon
        self.trackCon     = trackCon
        self.mpDraw       = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode,self.smooth,self.upper_body,self.detectionCon,self.trackCon)

    def findPose(self,img,draw = True):
        imgRgb = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        self.result = self.pose.process(imgRgb)
        if self.result.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img,self.result.pose_landmarks,self.mpPose.POSE_CONNECTIONS)
        
        return img    
    
    def getPose(self,img,draw=True):
        lmList = []
        
        if self.result.pose_landmarks:  
            for id, lm in enumerate(self.result.pose_landmarks.landmark):
                h, w, c= img.shape
                cx,cy= int(lm.x*w),int(lm.y*h)
                lmList.append((cx,cy,id))
                if draw:
                    cv.circle(img,(cx,cy),20,(255,50,30),1)
        return lmList

def main():
    cap = cv.VideoCapture("videos/dance1.mp4")
    ptime = 0
    detector = poseDetector()
    while True:
        success, img = cap.read()
        
        img = detector.findPose(img)
        lmList = detector.getPose(img)
        
        ctime = time.time()
        fps = 1/(ctime-ptime)
        ptime = ctime
        cv.putText(img,str(int(fps)),(100,50),cv.FONT_HERSHEY_SCRIPT_COMPLEX,2,(255,0,50),3)
        cv.imshow("Image",img)
        cv.waitKey(1)
    
if __name__=="__main__":
    main()