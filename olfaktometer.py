import numpy as np
import cv2
import time
import os
import shutil
import pandas as pd
import sys
from math import hypot
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
#aktiviert GPIOs als output channel
GPIO.setup(0, GPIO.OUT)#Ventil1
GPIO.setup(1, GPIO.OUT)#5
GPIO.setup(2, GPIO.OUT)#9
GPIO.setup(3, GPIO.OUT)#13
GPIO.setup(4, GPIO.OUT)#Ventil2
GPIO.setup(5, GPIO.OUT)#6
GPIO.setup(6, GPIO.OUT)#10
GPIO.setup(7, GPIO.OUT)#14
GPIO.setup(8, GPIO.OUT)#Ventil3
GPIO.setup(9, GPIO.OUT)#7
GPIO.setup(10, GPIO.OUT)#11
GPIO.setup(11, GPIO.OUT)#15
GPIO.setup(12, GPIO.OUT)#Ventil4
GPIO.setup(13, GPIO.OUT)#8
GPIO.setup(14, GPIO.OUT)#12
GPIO.setup(15, GPIO.OUT)#16
order=[0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15]#GPIO-Reihenfolge in der die Ventile aufgehen müssen

name=input('Name of the experiment: ')
path=('/home/pi/Experimente/'+name)
os.mkdir(path)
nproben=int(input('How many odors: '))
namen=[]
print('Lowest concentration first.')
for i in range(nproben):
    print('Name of odor', i+1)
    namen.append(input())
    print('Place odor at valve ', i+1)
messungen={}

class CoordinateStore:
    def __init__(self):
        super().__init__()
        self.points = []       
# mouse callback function
    def set_coordinates(self, event,x,y, flags,param): #Coordinates per doubleclick
        if event == cv2.EVENT_LBUTTONDBLCLK:
            global ix, iy
            ix, iy = x, y 
            self.points = (x,y)
            global click
            click = True  
#instantiate class
coordinateStore1 = CoordinateStore()

cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('path/to/recorded/video')
cv2.namedWindow('image')
cv2.setMouseCallback('image', coordinateStore1.set_coordinates)
#recording:
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('run.avi', fourcc, 5.0, (640,480))

armkoordinaten = [(118, 85), (151, 419), (494, 389), (458, 45)]

#shows coordinates of arena inlets
def showinlets(): 
    global frame, armkoordinaten
    cv2.circle(frame, (armkoordinaten[0][0], armkoordinaten[0][1]), 5, (0,0,255), -1)#Arm1
    cv2.circle(frame, (armkoordinaten[1][0], armkoordinaten[1][1]), 5, (0,0,255), -1)#Arm2
    cv2.circle(frame, (armkoordinaten[2][0], armkoordinaten[2][1]), 5, (0,0,255), -1)#Arm3
    cv2.circle(frame, (armkoordinaten[3][0], armkoordinaten[3][1]), 5, (0,0,255), -1)#Arm4

#Function to correct coordinates of arena exits
def armcorrect():
    global frame, ix, iy, click, armkoordinaten
    for i in range(4):
        print('Doubleclick on arena inlet ', i+1)
        while(1):
            ret, frame = cap.read()
            showinlets()
            if click == True:
                armkoordinaten[i]=(ix, iy)
                #print(armkoordinaten)
                click=False
                break
            cv2.imshow('image', frame)
            if cv2.waitKey(20) &0xFF ==27:
                break
    print('Are the displayed dots located precisely over the four arena inlets? (y/n)')

 #Function for measuring how much time the animal spends in which arm of arena       
def stoppuhr(): 
    global a1, a2, a3, a4, mitte, a1s, a2s, a3s, a4s, mitteS, zeit1, zeit2, armkoordinaten
    if hypot((x+75-armkoordinaten[0][0]), (y+75-armkoordinaten[0][1])) <= 175: #Arm nr. 1
        if a1s == False:
            zeit1=time.time()
            a1s=True
            a2s=False
            a3s=False
            a4s=False
            mitteS=False
        else:
            zeit2=time.time()
            a1+=(zeit2-zeit1)
            zeit1=time.time()
    elif hypot((x+75-armkoordinaten[1][0]), (y+75-armkoordinaten[1][1])) <= 175: #Arm nr. 2
        if a2s == False:
            zeit1=time.time()
            a1s=False
            a2s=True
            a3s=False
            a4s=False
            mitteS=False
        else:
            zeit2=time.time()
            a2+=(zeit2-zeit1)
            zeit1=time.time()
    elif hypot((x+75-armkoordinaten[2][0]), (y+75-armkoordinaten[2][1])) <= 175: #Arm nr. 3
        if a3s == False:
            zeit1=time.time()
            a1s=False
            a2s=False
            a3s=True
            a4s=False
            mitteS=False
        elif a3s== True:
            zeit2=time.time()
            a3+=(zeit2-zeit1)
            zeit1=time.time()
    elif hypot((x+75-armkoordinaten[3][0]), (y+75-armkoordinaten[3][1])) <= 175: #Arm nr. 4
        if a4s == False:
            zeit1=time.time()
            a1s=False
            a2s=False
            a3s=False
            a4s=True
            mitteS=False
        else:
            zeit2=time.time()
            a4+=(zeit2-zeit1)
            zeit1=time.time()
    else: #mitte
        if mitteS == False:
            zeit1=time.time()
            a1s=False
            a2s=False
            a3s=False
            a4s=False
            mitteS=True
        else:
            zeit2=time.time()
            mitte+=(zeit2-zeit1)
            zeit1=time.time()
            
def resetstoppuhr():
    global a1, a2, a3, a4, mitte, a1s, a2s, a3s, a4s, mitteS
    a1=0#Times
    a2=0
    a3=0
    a4=0
    mitte=0
    a1s=False#PositiPositionsonen
    a2s=False
    a3s=False
    a4s=False
    mitteS=False

#olfaction experiment function            
def olfa():
    global zeitR, rundenZeit, clean, runde, R, a1, a2, a3, a4, mitte, a1s, a2s, a3s, a4s, mitteS, rimage
    if clean==True:
        GPIO.output(order[runde-1], GPIO.HIGH)#open valve
        rundenZeit=time.time()-zeitR
        rimage=cv2.putText(frame, R, (17,444), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)
        stoppuhr()
        if rundenZeit>=30:
            GPIO.output(order[runde-1], GPIO.LOW)#close valve
            clean=False
            zeitR=time.time()
            messungen[namen[runde-1]]=[a1, a2, a3, a4, mitte]
    else:
        rundenZeit=time.time()-zeitR
        rimage=cv2.putText(frame, 'Spuelung', (17,444), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)
        if rundenZeit>=10:
            clean=True
            runde+=1
            R='Runde '+str(runde)
            resetstoppuhr()
            zeitR=time.time()

#Video recording function
def record():
    global rimage
    if ret==True:
        out.write(rimage)
        
def timer():
    global zeit, elapsedZeit, rimage, frame
    elapsedZeit=time.time()-zeit
    timeS=format(elapsedZeit, '.2f')
    rimage=cv2.putText(frame, timeS, (17,464), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)

#cancel function     
def abbruch():
    cap.release()
    cv2.destroyAllWindows()
    print('Experiment cancelled')
    sys.exit()

#Modifying image data for tracking
def bildverarbeitung(): 
    global frame, blur, fgmask, fgbg
    fgmask = fgbg.apply(frame)
    #unscharf
    blur = cv2.GaussianBlur(fgmask, (13, 13), 0)

#Backgroundsubtraction
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

#Trackingfunction      
def track(): 
    global frame, blur, track_window, term_crit, ret, x, y
    if ret == True:
        # apply meanshift to get the new location
        ret, track_window = cv2.meanShift(blur, track_window, term_crit)
        # Draw it on image
        x,y,w,h = track_window
        img1 = cv2.rectangle(frame, (x,y), (x+h, y+w), 255,2)
        img2 = cv2.circle(frame, (x+73, y+73), 1, (-1,255,0), 1)
 
def showarms():
    global frame, armkoordinaten
    cv2.circle(frame, (armkoordinaten[0][0], armkoordinaten[0][1]), 175, (0,0,255), 1)
    cv2.circle(frame, (armkoordinaten[0][0], armkoordinaten[0][1]), 5, (0,0,255), -1)
    cv2.circle(frame, (armkoordinaten[1][0], armkoordinaten[1][1]), 175, (0,0,255), 1)
    cv2.circle(frame, (armkoordinaten[1][0], armkoordinaten[1][1]), 5, (0,0,255), -1)
    cv2.circle(frame, (armkoordinaten[2][0], armkoordinaten[2][1]), 175, (0,0,255), 1)
    cv2.circle(frame, (armkoordinaten[2][0], armkoordinaten[2][1]), 5, (0,0,255), -1)
    cv2.circle(frame, (armkoordinaten[3][0], armkoordinaten[3][1]), 175, (0,0,255), 1)
    cv2.circle(frame, (armkoordinaten[3][0], armkoordinaten[3][1]), 5, (0,0,255), -1) 

click = False # = no doubleclick detected yet
#correct arena inlet coordinates if necessary
print('Are the displayed dots located precisely over the four arena inlets? (y/n)')
while(1):
    ret, frame = cap.read()
    showarms()
    cv2.imshow('image', frame)
    if cv2.waitKey(20) == ord('y') or cv2.waitKey(20) &0xFF==27:
        print('Coordinates correct')
        print(armkoordinaten)
        break
    elif cv2.waitKey(30) == ord('n'):
        print('Reset arena arm exit coordinates')
        armcorrect()    
cap.release
cv2.destroyAllWindows

pixelprometer = hypot(armkoordinaten[0][0]-armkoordinaten[2][0], armkoordinaten[0][1]-armkoordinaten[2][1])*(10/3) #30=distance between exits

#open videostream to mark animal
print('Doubleclick on animal to start experiment')
while(1):
    ret, frame = cap.read()
    cv2.imshow('image', frame)
    if click == True: #when doubleclick or esc -> break
        break
    elif cv2.waitKey(20) &0xFF ==27:
        abbruch()
cv2.destroyAllWindows()

# setup initial location of tracking window
r,h,c,w = iy-75,150,ix-75,150  #doubleclick coordinates
track_window = (c,r,w,h)
# set up the ROI for tracking
roi = frame[r:r+h, c:c+w]
hsv_roi =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

#Agility
zeit = time.time()
elapsedZeit = 0
path = 0
xalt = ix
yalt = iy

resetstoppuhr()

while(1):
    ret, frame = cap.read()
    bildverarbeitung()
    track()
    path += hypot(x-xalt, y-yalt)
    xalt=x
    yalt=y
    stoppuhr()
    timer()
    rimage=cv2.putText(frame, 'Agility', (17,444), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
    record()
    cv2.imshow('Agility', frame)
    if cv2.waitKey(20) & 0xFF == 27:
        abbruch()
    if elapsedZeit >= 120:
        ag=[path/pixelprometer, elapsedZeit, path/pixelprometer/elapsedZeit]
        agtable=pd.DataFrame(ag , columns = ['1'], index = ['Travelled distance[m]', 'Runtime[s]', 'Average speed[m/s]'])
        print(agtable)
        agtable.to_csv('agility.csv', sep=';', decimal=',')
        shutil.move('/home/pi/agility.csv', ('/home/pi/Experimente/'+name))
        praef=[(a1/elapsedZeit)*100, (a2/elapsedZeit)*100, (a3/elapsedZeit)*100, (a4/elapsedZeit)*100, (mitte/elapsedZeit)*100]
        praefTable=pd.DataFrame(praef , columns = ['% von 120s'], index = ['Arm 1', 'Arm 2', 'Arm 3', 'Arm 4', 'Middle'])
        print(praefTable)
        praefTable.to_csv('preference.csv', sep=';', decimal=',')
        shutil.move('/home/pi/preference.csv', ('/home/pi/Experimente/'+name))
        break
cv2.destroyAllWindows()

zeit=time.time()#for total time
zeitR=time.time()#for time of one round
clean=True
runde=1
R='Round '+str(runde)
resetstoppuhr()

#Experiment:
while(1):
    ret, frame = cap.read()
    bildverarbeitung()
    track()
    olfa()
    timer()
    showarms()
    record()
    cv2.imshow('Experiment', frame)
    if runde >= nproben+1 or cv2.waitKey(20) & 0xFF == 27:
        print(messungen)
        dfmessungen=pd.DataFrame(data=messungen, index=['Arm 1', 'Arm 2', 'Arm 3', 'Arm 4', 'Mitte'])
        print(dfmessungen)
        dfmessungen.to_csv('Olfaktionsexperiment.csv', sep=';', decimal=',')
        shutil.move('/home/pi/Olfaktionsexperiment.csv', ('/home/pi/Experimente/'+name))
        break


shutil.move('/home/pi/run.avi', ('/home/pi/Experimente/'+name))
cap.release()
out.release()
cv2.destroyAllWindows()
