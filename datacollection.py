import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
cap=cv2.VideoCapture(0) #0 for device camera and 1 for web camera
detector=HandDetector(maxHands=1) #Only 1 hand will be detected at a time
offset=20
imgSize=200
counter=0
folder="/Users/macbook/Desktop/Sign Language Detection/Data/I Love You"
while True:
    success,img=cap.read()
    hands,img=detector.findHands(img) #Will find hands in the image
    if hands:
        hand=hands[0]
        x,y,w,h =hand['bbox'] #x-axis, y-axis, width, height
        imgWhite=np.ones((imgSize,imgSize,3),np.uint8)*255 #for white bg of hand that is detected
        imgCrop=img[y-offset:y+h+offset,x-offset:x+w+offset]
        imgCropShape=imgCrop.shape

        aspectratio=h/w
        if aspectratio>1:
            k=imgSize/h
            wCal=math.ceil(k*w)
            imgResize=cv2.resize(imgCrop,(wCal,imgSize))
            imgResizeShape=imgResize.shape
            wGap=math.ceil((imgSize-wCal)/2)
            imgWhite[: ,wGap : wCal + wGap]=imgResize

        else:
            k=imgSize/w
            hCal=math.ceil(k*h)
            imgResize=cv2.resize(imgCrop,(imgSize,hCal))
            imgResizeShape=imgResize.shape
            hGap=math.ceil((imgSize-hCal)/2)
            imgWhite[hGap : hCal + hGap , : ]=imgResize

        cv2.imshow('ImageCrop',imgCrop)
        cv2.imshow('ImageWhite',imgWhite)
    
    cv2.imshow("Image",img)
    key=cv2.waitKey(1) #Key on keyboard for collecting data
    if key==ord('s'):
        counter+=1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
        print(counter)

