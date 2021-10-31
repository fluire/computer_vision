import cv2 as cv
import mediapipe as mp
import time

class Facemeshdetector:
    def __init__(self,staticMode = False,max_faces = 3,min_detection_confidence = 0.5,min_tracking_confidence = 0.5):
        self.staticMode = staticMode
        self.max_faces = max_faces
        self.min_detection_con = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
         
        self.mpDraw = mp.solutions.drawing_utils 
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode,self.max_faces
                                                 ,self.min_detection_con,self.min_tracking_confidence)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)


    def findFaceMesh(self,img,draw = True):
        self.imgRgb = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRgb)
        faces = []
        if self.results.multi_face_landmarks:
            
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,faceLms,self.mpFaceMesh.FACE_CONNECTIONS,
                                    self.drawSpec,self.drawSpec)
                face = []
                for id,lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x,y= int(iw*lm.x),int(ih*lm.y)
                    #cv.putText(img,str(id),(x,y),cv.Formatter_FMT_C,0.5,(0,250,0),1)    

                    #print(id,x,y)
                    face.append([x,y])
                faces.append(face)
                    
        return img,faces
    
    

def main():
    cap = cv.VideoCapture("videos/face2.mp4")
    ptime= 0 
    detector = Facemeshdetector()
    while True:
        success,img = cap.read()
        img,faces = detector.findFaceMesh(img)
        if len(faces)!=0:
            print(faces[0])
        ctime = time.time()
        fps = int(1/(ctime-ptime))
        ptime = ctime
        cv.putText(img,f'FPS:{fps}',(20,70),cv.Formatter_FMT_C,1,(0,255,0),3)    
        cv.imshow("facemesh",img)
        cv.waitKey(1)
    
if __name__ == "__main__":
    main()