# -*- coding: cp1252 -*-
import numpy as np
import cv2
import time
from time import sleep
import os, sys
from threading import Thread 

""" Simulate the searching process into the photo database """
def simulate(X):
    while 1:
        for zz in range(1,len(X)/2):
            image = cv2.resize(X[zz*2],(150,150))
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            cv2.putText(image,'searching...',(10, 20), cv2.FONT_HERSHEY_PLAIN,1.3,(0, 255, 0))
            cv2.imshow("DB",image)
            cv2.waitKey(70)


def read_images(path, image_size=(100,100) ):
    """ Reads the images in a given folder

    Args:
        path: Path to a folder with subfolders representing the subjects (persons).
        size: A tuple with the size 

    Returns:
        A list [X, y, folder_names, gender_vector]

            X: The images, which is a Python list of numpy arrays.
            y: The corresponding labels (the unique number of the subject, person) in a Python list.
            folder_names: The names of the folder, so you can display it in a prediction.
            gender_vector: the gender of the person

    Database structure example:

    database
    |
    |----> andrea _ m
    |        |---> 1.jpg
    |        |---> 2.jpg
    |        |---> 3.jpg
    |
    |----> chiara _ f
    |        |
    |        |---> 1.jpg
    |        |---> 2.jpg
    |---->
    |
    """
    c = 0
    X = [] # person pic
    y = [] # label
    gender_vector = [] # person gender
    folder_names = [] # person name  
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            try:
                # subdirname is the name of the folder in which are stored photos. ( example: andrea _ m )
                name, gender = subdirname.split("_")
            except:
                gender = "None"
            folder_names.append(name)
            gender_vector.append(gender)
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                try:
                    im = cv2.imread(os.path.join(subject_path, filename))
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                    im = cv2.equalizeHist(im)
                    if (image_size is not None):
                        im = cv2.resize(im, image_size)
                    X.append(np.asarray(im, dtype=np.uint8))
                    y.append(c)
                except IOError, (errno, strerror):
                    print "I/O error({0}): {1}".format(errno, strerror)
                except:
                    #None
                    print "[-] Error opening "+filename
                    
            c = c+1
    return [X,y,folder_names,gender_vector]


""" Print all faces found in a new window """
def showManyImages(title, faccia):
    pics = len(faccia) # number of found faces
    # r - Maximum number of images in a row 
    # c - Maximum number of images in a column 
    '''if pics < 1 or pics > 12:
        return
    elif pics == 1:
        r = 1
        c = 1
        size = (250, 250)
    elif pics == 2:
        r = 2
        c = 1
        size = (150, 150)
    elif pics == 3 or pics == 4:
        r = 2
        c = 2
        size = (100, 100)
    elif pics == 5 or pics == 6:
        r = 3
        c = 2
        size = (100, 100)
    elif pics == 7 or pics == 8:
        r = 4
        c = 2
        size = (75, 75)
    else:
        r = 4
        c = 3
        size = (60, 60)'''

    ##################
    r = 4
    c = 2
    size = (100, 100)
    ##################
            
    width = size[0]
    height = size[1]
    #print "Creating %s-%s matrix"%(height*r, width*c)
    image = np.zeros((height*r, width*c, 3), np.uint8)
    black = (0, 0, 0) # background color
    rgb_color = black
    color = tuple(reversed(rgb_color))
    image[:] = color


    for i in range(0,len(faccia)):
        faccia[i] = cv2.resize(faccia[i], size )
        cv2.rectangle(faccia[i],(0,0), size,(0,0,0),4) # face edge

    k=0
    for i in range(0, r):
        for j in range(0, c):
            #print "\tWriting into "+str(i*size[0])+" "+str(i*size[0]+size[0])+" - "+str(j*size[1])+" "+str(j*size[1]+size[1])
            try:
                image[i*size[0]:i*size[0]+size[0], j*size[0]:j*size[0]+size[0]] = faccia[k]
            except:
                None
            k=k+1
        
    cv2.imshow(title,image)


    
####################################################################
#                        MAIN
####################################################################


# read images in the folder
[X,y,folder_names, gender_vector] = read_images("database") 
 
y = np.asarray(y, dtype=np.int32)

# Create the model
#model = cv2.createEigenFaceRecognizer()
model = cv2.createLBPHFaceRecognizer()
#model = cv2.createFisherFaceRecognizer()

# Learn the model
print "[*] Training..."
try:
    model.train(np.asarray(X), np.asarray(y))
    print "[+] Done"
    print "[*] Starting detection..."
except:
    print "[-] Fail"
    sys.exit()

face_cascade = cv2.CascadeClassifier('C:\opencv\data\haarcascades\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:\opencv\data\haarcascades\haarcascade_eye.xml')
#eye_cascade_glasses = cv2.CascadeClassifier('C:\opencv\data\haarcascades\haarcascade_eye_tree_eyeglasses.xml')
mouth_cascade = cv2.CascadeClassifier('C:\opencv\data\haarcascades\haarcascade_mcs_mouth.xml')
nose_cascade = cv2.CascadeClassifier('C:\opencv\data\haarcascades\haarcascade_mcs_nose.xml')

webcam = cv2.VideoCapture(0)
#webcam.set(3,1280)
#webcam.set(4,720)
size=2

# generate the thread "simulate" as deamon to close it when the main die
thread = Thread(target=simulate, args=[np.asarray(X)])
thread.daemon = True
thread.start()

start = time.time()
count = 0

while True:
    
    (rval, frame) = webcam.read()
 
    height, width, depth = frame.shape
    # if the width is too high, resize it for faster processing
    if width > 640:
        frame = cv2.resize(frame,(1024,600))
        size = 3
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    mini = cv2.resize(gray, (gray.shape[1] / size, gray.shape[0] / size))


    #faces = face_cascade.detectMultiScale(mini, scaleFactor=1.2, minNeighbors=4, minSize=(10, 10), maxSize=(600, 600))
    #faces = face_cascade.detectMultiScale(mini,minNeighbors=4,minSize=(10, 10))
    faces = face_cascade.detectMultiScale(mini)
    faccia=[] # used to collect found faces
    
    for i in range(len(faces)):
        face_i = faces[i]
        (x, y, w, h) = [v * size for v in face_i]
        face = gray[y:y+h, x:x+w]

        width, height = face.shape
        
        # calcolo distanza
        # d = width, Z = real distance, D = real width 
        # f = d*Z/D
        #
        # Z' = distanza da misurare, d' = width
        # Z' = D*f/d'
        #        
        f = 202*52/18
        distance = 20*f/width
        
        face_resize = cv2.resize(face, (100,100))
        face_color = frame[y:y+h, x:x+w]
        face_color2 = face_color.copy()

        roi_eye_gray = face[height/5:height*6/11 , width*1/10:width*9/10] 
        roi_eye_color = face_color[height/5:height*6/11 , width*1/10:width*9/10] 
    
        roi_nose_gray = face[height*2/5:height*8/11 , width*2/7:width*5/7]
        roi_nose_color = face_color[height*2/5:height*8/11 , width*2/7:width*5/7]

        roi_mouth_gray = face[height*7/11:height , width*1/6:width*5/6]
        roi_mouth_color = face_color[height*7/11:height , width*1/6:width*5/6]

        #cv2.imshow('roi1',roi_eye_gray)
        #cv2.imshow('roi2',roi_nose_gray)
        #cv2.imshow('roi3',roi_mouth_gray)

        eyes = []
        eyes = eye_cascade.detectMultiScale(roi_eye_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_eye_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),1)
        #eyes_g = []
        #eyes_g = eye_cascade_glasses.detectMultiScale(roi_eye_gray)
        #for (ex,ey,ew,eh) in eyes_g:
            #cv2.rectangle(roi_eye_color,(ex,ey),(ex+ew,ey+eh),(0,255,255),2)
        mouth = []
        mouth = mouth_cascade.detectMultiScale(roi_mouth_gray)
        for (ex,ey,ew,eh) in mouth:
            cv2.rectangle(roi_mouth_color,(ex,ey),(ex+ew,ey+eh),(255,255,0),1)
        nose = []
        nose = nose_cascade.detectMultiScale(roi_nose_gray)
        for (ex,ey,ew,eh) in nose:
            cv2.rectangle(roi_nose_color,(ex,ey),(ex+ew,ey+eh),(255,0,0),1)


        if len(eyes) != 0 or len(mouth) != 0 or len(nose) != 0 or width < 65:   
            faccia.append(face_color2)
            prediction = model.predict(face_resize)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
            if prediction[1] < 120:
                cv2.putText(frame,'%s - %.0f' % (folder_names[prediction[0]],prediction[1]),(x, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
                cv2.putText(frame,'ID: %d' % (prediction[0]),(x, y+h+15), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))

                if gender_vector[prediction[0]].strip() == "m":
                    gender = "Male"
                elif gender_vector[prediction[0]].strip() == "f":
                    gender = "Female"

                cv2.putText(frame,'Gender: %s' % gender,(x, y+h+30), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
                cv2.putText(frame,'Distance: %d cm' % (distance),(x, y+h+45), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
            else:
                cv2.putText(frame,'Unknown',(x, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 0, 255))
                cv2.putText(frame,'Distance: %d cm' % (distance),(x, y+h+15), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))

    count = count+1    
    current = time.time()
    cv2.putText(frame, " fps=%.1f" % (count/(current-start)),(0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0)) # frame processed per second
    
    cv2.imshow('Video Stream', frame)
    showManyImages("Found",faccia)
    key = cv2.waitKey(1)
    if key == 27:
        break


cap.release()

