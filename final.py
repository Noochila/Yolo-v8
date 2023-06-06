import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import os
from tracker import *

model_path = os.path.join('.', 'runs', 'detect', 'train2', 'weights', 'best.pt')
model=YOLO(model_path)

d='';

area=[(303,316),(815,327),(727,473),(4,402)]  
area_c1=set()
area_c2=set()
area_c3=set()
area_c4=set()
tracker=Tracker()


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)


cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap=cv2.VideoCapture('videos/IMG_5008.mp4')

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

count=0
while True:
    
    ret,frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame=cv2.resize(frame,(1020,500))

    results=model.predict(frame)

    a=results[0].boxes.boxes
    px=pd.DataFrame(a).astype(float)
    list1=[]
    list2=[]
    list3=[]
    list4=[]

    for index,row in px.iterrows():
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        if d==0:
            list1.append([x1,y1,x2,y2])
        elif d==1:
            list2.append([x1,y1,x2,y2])
        elif d==2:
            list3.append([x1,y1,x2,y2])
        elif d==3:
            list4.append([x1,y1,x2,y2])
    if d==0:
        bbox1_id=tracker.update(list1)
        for bbox1 in bbox1_id:
            x3,y3,x4,y4,id=bbox1
            cx=int(x3+x4)//2
            cy=int(y3+y4)//2
            result=cv2.pointPolygonTest(np.array(area,np.int32),(cx,cy),False)
            if result>=0:
                cv2.rectangle(frame,(x3,y4),(x4,y4),(0,225,0),3)
                cv2.putText(frame,str(c),(x3,y3),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,225),2)
                cv2.circle(frame,(cx,cy),4,(0,0,225),-1)
                area_c1.add(id)
    elif d==1:            
        bbox2_id=tracker.update(list2)
        for bbox2 in bbox2_id:
            x3,y3,x4,y4,id=bbox2
            cx=int(x3+x4)//2
            cy=int(y3+y4)//2
            result=cv2.pointPolygonTest(np.array(area,np.int32),(cx,cy),False)
            if result>=0:
                cv2.rectangle(frame,(x3,y4),(x4,y4),(0,225,0),3)
                cv2.putText(frame,str(c),(x3,y3),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,225),2)
                cv2.circle(frame,(cx,cy),4,(0,0,225),-1)
                area_c2.add(id)
    elif d==2:              
        bbox3_id=tracker.update(list3)
        for bbox3 in bbox3_id:
            x3,y3,x4,y4,id=bbox3
            cx=int(x3+x4)//2
            cy=int(y3+y4)//2
            result=cv2.pointPolygonTest(np.array(area,np.int32),(cx,cy),False)
            if result>=0:
                cv2.rectangle(frame,(x3,y4),(x4,y4),(0,225,0),3)
                cv2.putText(frame,str(c),(x3,y3),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,225),2)
                cv2.circle(frame,(cx,cy),4,(0,0,225),-1)
                area_c3.add(id)
    else:            
        bbox4_id=tracker.update(list4)
        for bbox4 in bbox4_id:
            x3,y3,x4,y4,id=bbox4
            cx=int(x3+x4)//2
            cy=int(y3+y4)//2
            result=cv2.pointPolygonTest(np.array(area,np.int32),(cx,cy),False)
            if result>=0:
                cv2.rectangle(frame,(x3,y4),(x4,y4),(0,225,0),3)
                cv2.putText(frame,str(c),(x3,y3),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,225),2)
                cv2.circle(frame,(cx,cy),4,(0,0,225),-1)
                area_c4.add(id)            

    cv2.putText(frame,"Two wheeler(Bikes,Scooter,Cycle)" + str(len(area_c1)),(47,56),cv2.FONT_HERSHEY_PLAIN,2,(0,225,0),2)
    cv2.putText(frame,"Auto(Three Wheeler)"+ str(len(area_c2)),(40,159),cv2.FONT_HERSHEY_PLAIN,2,(0,225,0),2)
    cv2.putText(frame,'Four Wheeler(Cars,Mini-Trucks,Taxi)'+ str(len(area_c3)),(39,260),cv2.FONT_HERSHEY_PLAIN,2,(0,225,0),2)
    cv2.putText(frame,'Large Vehicles(Lorry,Trucks,Tanker)'+ str(len(area_c4)),(49,448),cv2.FONT_HERSHEY_PLAIN,2,(0,225,0),2)        
    print(area_c1)     
    print(area_c2)     
    print(area_c3)     
    print(area_c4)     
    cv2.polylines(frame,[np.array(area,np.int32)],True,(0,125,0),3)
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break

cap.release()
cv2.destroyAllWindows()

            