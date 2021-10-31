import cv2 as cv
import mediapipe as mp
import time

#class for the use of face detection and drawing landmarks

class FaceDetector:
    def __init__(self, minDetectionCon=0.5):
        self.minDetectionCon = minDetectionCon    
        self.mpDraw = mp.solutions.drawing_utils
        self.face = mp.solutions.face_detection
        self.detFace = self.face.FaceDetection(self.minDetectionCon)

    def findFaces(self,img,draw=True):
        imgRgb = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        self.result = self.detFace.process(imgRgb)
        bboxes = []
        if self.result.detections:
            for id,detection in enumerate(self.result.detections):
                #mpDraw.draw_detection(img,detection)
                ih, iw, ic = img.shape
                bboxC = detection.location_data.relative_bounding_box
                bbox = int(bboxC.xmin*iw),int(bboxC.ymin*ih),\
                    int(bboxC.width*iw),int(bboxC.height*ih)
                bboxes.append([id,bbox,detection.score])
                if draw:
                    self.fancyDraw(img,bbox)
                    cv.putText(img,f'{int(detection.score[0]*100)}',(bbox[0],bbox[1]-20),cv.FONT_HERSHEY_PLAIN,2,(0,200,0),2)
        return img,bboxes
    
    def fancyDraw(self,img,bbox,l=30,w=30,t=10,rt=1):
        x, y, width, height  = bbox
        x1,y1 = x+width,y+height
        cv.rectangle(img,bbox,(255,0,255),rt)
        #top left
        cv.line(img, (x,y),(x+l,y),(255,0,255),t)
        cv.line(img, (x,y),(x,y+w),(255,0,255),t)
        #bottom right
        cv.line(img, (x1,y1),(x1-l,y1),(255,0,255),t)
        cv.line(img, (x1,y1),(x1,y1-w),(255,0,255),t)
        #top right
        cv.line(img, (x1,y),(x1-l,y),(255,0,255),t)
        cv.line(img, (x1,y),(x1,y+w),(255,0,255),t)
        #bottom left
        cv.line(img, (x,y1),(x+l,y1),(255,0,255),t)
        cv.line(img, (x,y1),(x,y1-w),(255,0,255),t)
        

            
   
    
def main():
    cap = cv.VideoCapture(0)
    ptime = 0
    detection = FaceDetector(0.75)
    while True:
        success, img = cap.read()
        img,bboxes = detection.findFaces(img)
        if len(bboxes)>0:
            print(bboxes)
        ctime = time.time()
        fps = int(1/(ctime-ptime))
        ptime = ctime
        cv.putText(img,f'FPS:{fps}',(50,70),cv.FONT_HERSHEY_PLAIN,2,(200,42,0),2)
        cv.imshow("face",img)
        cv.waitKey(1)
    
    
if __name__ == "__main__":
    main()
