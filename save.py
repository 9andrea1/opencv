import numpy as np
import cv2
import time
from time import sleep
import os, sys

#from Tkinter import Tk
#from tkFileDialog import askdirectory
#Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
#dirname = askdirectory(title='Select a directory')
#print dirname


start = time.time()
count = 0

k = 0

face_cascade = cv2.CascadeClassifier('C:\opencv\data\haarcascades\haarcascade_frontalface_default.xml')

webcam = cv2.VideoCapture(0)
size=1


while True:

    count = count+1    
    current = time.time()
    
    (rval, frame) = webcam.read()

    width, height, depth = frame.shape
    print str(width)+" "+str(height)+" "+str(depth)


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #gray = cv2.equalizeHist(gray)

    mini = cv2.resize(gray, (gray.shape[1] / size, gray.shape[0] / size))

    width, height = mini.shape
    print str(width)+" "+str(height)
    

    #faces = face_cascade.detectMultiScale(mini, scaleFactor=1.2, minNeighbors=4, minSize=(10, 10), maxSize=(600, 600))
    faces = face_cascade.detectMultiScale(mini)
    faccia=[]
    print "found %s faces"%len(faces)
    for i in range(len(faces)):
        face_i = faces[i]
        (x, y, w, h) = [v * size for v in face_i]
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (100,100))
        face_color = frame[y:y+h, x:x+w]
        if( cv2.imwrite("C:\\Users\\Andrea\\Desktop\\face\\database\\found\\face-"+str(k)+".jpg",face_color) ):
            k = k+1
            cv2.imshow("Saving...",face_color)
            print "face-%d.jpg saved"%(k-1)
        sleep(1)
        
    
    cv2.imshow('Video Stream', frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

    
      
cap.release()

