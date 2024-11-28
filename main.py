import tkinter as tk
import os
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

model = YOLO("best.pt")
classifier = load_model(os.path.join('models','vgg19_v2.h5'))
fileName = os.environ['ALLUSERSPROFILE'] + "\WebcamCap.txt"
cancel = False
    
def changeCam(event=0, nextCam=-1):
    global camIndex, cap, fileName

    if nextCam == -1:
        camIndex += 1
    else:
        camIndex = nextCam
    del(cap)
    cap = cv2.VideoCapture(camIndex)

    #try to get a frame, if it returns nothing
    success, frame = cap.read()
    if not success:
        camIndex = 0
        del(cap)
        cap = cv2.VideoCapture(camIndex)

try:
    f = open(fileName, 'r')
    camIndex = int(f.readline())
except:
    camIndex = 0

cap = cv2.VideoCapture(camIndex)

success, frame = cap.read()


mainWindow = tk.Tk(screenName="Camera Capture")
lmain = tk.Label(mainWindow, compound=tk.CENTER, anchor=tk.CENTER, relief=tk.RAISED)
button_changeCam = tk.Button(mainWindow, text="Switch Camera", command=changeCam)

lmain.pack()
button_changeCam.place(bordermode=tk.INSIDE, relx=0.85, rely=0.1, anchor=tk.CENTER, width=150, height=50)

def show_frame():
    global cancel, prevImg, button, temp
    temp = 0
    _, frame = cap.read()
    cv2image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    result = model(cv2image)
    result = result[0].boxes
    for box in result:
        top_left_x = int(box.xyxy.tolist()[0][0])
        top_left_y = int(box.xyxy.tolist()[0][1])
        bottom_right_x = int(box.xyxy.tolist()[0][2])
        bottom_right_y = int(box.xyxy.tolist()[0][3])
        detect = tf.image.resize(frame[top_left_y:bottom_right_y,top_left_x:bottom_right_x],(224,224))
        detect = np.reshape(detect,[1,224,224,3])
        predict = classifier(detect)
        classes_x = np.argmax(predict)
        if classes_x==1:
            cv2.putText(frame,"Tidak Mengantuk",(top_left_x,top_left_y-10), cv2.FONT_HERSHEY_DUPLEX,0.6,(0,255,0))
            cv2.rectangle(frame,(top_left_x,top_left_y),(bottom_right_x,bottom_right_y),(0,255,0),2)
        else:
            cv2.putText(frame,"Mengantuk",(top_left_x,top_left_y-10), cv2.FONT_HERSHEY_DUPLEX,0.6,(0,0,255))
            cv2.rectangle(frame,(top_left_x,top_left_y),(bottom_right_x,bottom_right_y),(0,0,255),2)
            temp = temp + 1

    cv2.putText(frame,"Jumlah Wajah Mengantuk :"+str(temp), (40,40), cv2.FONT_HERSHEY_DUPLEX,0.7,(153,0,0))
    prevImg = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=prevImg)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    if not cancel:
        lmain.after(10, show_frame)

show_frame()
mainWindow.mainloop()
