import keras
from keras.models import load_model
import cv2
import numpy as np
import imutils
import operator

model=load_model("C:/Users/ELCOT/Desktop/Mask/model.h5")

cap = cv2.VideoCapture(0)

categories={0:"No Mask",1:"Mask Detected",2:'nothing'}

while True:
    _,frame=cap.read()
    frame = cv2.flip(frame,1)
    x1=int(0.2*frame.shape[1])
    y1=10
    x2=frame.shape[1]-100
    y2=int(0.6*frame.shape[1])

    cv2.rectangle(frame,(x1-1,y1-1),(x2-1,y2-1),(255,0,0),3) 
    roi=frame[y1:y2,x1:x2]
    roi= cv2.resize(roi,(50,50))

    result= model.predict(roi.reshape(1,50,50,3))

    prediction={
                 "No Mask":result[0][0],
                 "Mask Detected":result[0][1],
                 'Nothing':result[0][2]
                 }

    prediction=sorted(prediction.items(),key=operator.itemgetter(1),reverse=True) 
    cv2.putText(frame,prediction[0][0],(x1+100,y2+30),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),3)
    cv2.imshow("frame",frame)
    print(prediction[0][0])

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

    
    
