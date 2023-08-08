import cv2 as cv
import numpy as np
#import pytesseract py tesseract not good for this application
import easyocr
from PIL import Image, ImageTk
import pandas as pd
from datetime import datetime
import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog import askopenfile
import csv
import time
from threading import *
testingmode = False
global status
#rotate90 = False# not implemented

#pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract'
orangenumbers = [
['1','I','T','['],#1 robust
['2','N'],#2
['3','E','W','M','€'],#3
['4'],#4
['5','S','z'],#5 Robust
['6','9',],#6
['7','L','l'],#7 robust
['0','o','O']]#H20
whitenumbers = [
['1','I','T','['],#1 robust
['2','N'],#2
['3','E','W','M','€'],#3
['4'],#4
['5','S','z'],#5 Robust
['6','9',],#6
['7','L','l'],#7 robust
[],#10
['H','h','8']]#10

boundingboxes = [(35, 35, 131, 116, 0, 0), (175, 33, 137, 118, 0, 1), (329, 33, 133, 120, 0, 2), (488, 38, 127, 116, 0, 3), (32, 178, 129, 116, 1, 0), (177, 176, 130, 117, 1, 1), (332, 176, 131, 118, 1, 2), (485, 180, 127, 116, 1, 3), (32, 322, 118, 115, 2, 0), (181, 320, 120, 117, 2, 1), (330, 325, 131, 115, 2, 2), (486, 327, 129, 114, 2, 3), (32, 467, 115, 115, 3, 0), (177, 468, 131, 115, 3, 1), (331, 467, 137, 117, 3, 2), (484, 470, 129, 116, 3, 3), (32, 615, 120, 114, 4, 0), (177, 616, 133, 113, 4, 1), (331, 617, 129, 114, 4, 2), (485, 620, 130, 113, 4, 3), (32, 764, 129, 111, 5, 0), (181, 765, 135, 113, 5, 1), (334, 765, 127, 113, 5, 2), (486, 765, 130, 114, 5, 3)]

doubleweight = ['H','h','S','5','E','w','m','2','W','3','4','O']
finalnumbers = ['1','2','3','4','5','6','7','0','h',]#scuffed code haven't included H2O yet
colorkey={'o':'orange','w':'white','r':'red','b':'gray30','p':'purple','g':'green','h':'gray70','y':'yellow','e':'gray60'}#b means blank, e means empty
numberkey={'1':'1','2':'2','3':'3','4':'4','5':'5','6':'6','9':'6','7':'7','0':'10','h':'H','b':' ','r':' ','g':' ','p':' ','y':' ','e':' ',' ':' '}
letkey = ['A','B','C','D']
numkey = ['1','2','3','4','5','6']
#snapy = [0, 150, 300, 450, 600, 750]#bbounding boxes snap to the 4x6 grid
#snapx = [0, 150, 300, 450]#only for calibration
f = open('roi.txt', 'r')
roi = f.read().split('\n')
for i in range(4):
    roi[i] = tuple(map(int,roi[i].split(' ')))
f.close()
#calibratedcorners = roi#[(714,476),(1282,464),(1242,1344),(670,1336)]#top left top right bottom right bottom left
colorbound = [#all of this is in BGR,  lower upper threshold for color
((1,0.7,0),(1.2,0.9,0.8),'p'),#purple 64
((1.3,1.3,0),(2.5,2.5,10),'r'),#red 57
((0.6,0.7,0),(0.9,1.2,10),'g'),#green  54
((0.9,1.3,0),(1.2,2.0,10),'y'),#yellow
((1.1,1,0),(1.3,1.3,10),'o'),#orange 8.4  6.8 18
((0,0,0),(100,100,100),'w')]#gray 26 20
currentpoint = ()
importtube = []
tubeinfo = []
tkimporttubes = []
def camwarmup():
    global cam

    cam = cv.VideoCapture(0)
    ret,_ = cam.read()
    time.sleep(3)#camera gotta warm up
    print (cam)

if testingmode == False:
    t0 = Thread(target=camwarmup)#I don't know why but this fixes a lighting issue getting a single frame will make the image really dark (bad)
    t0.start()
reader = easyocr.Reader(['en'])#maybe thread this later who knows
#if ret == False: 
#    print ('ERROR NO CAMERA ATTACHED')
#print (ret)
def ocr(cap,accuracy):

    charlist = ''

    for i in range(accuracy):#change value to increase or decrease accuracy
        new = rotateImage(cap,360/accuracy*i)
        #cv.imshow("Canny Edged", new)
        #cv.waitKey(0)
        char =(reader.readtext(new, detail = 0,batch_size=16, ))#allowlist='1234567890oOLTISEMW'))
        if char != []:
            char = (char[0])
            charlist += char
    return charlist
def uncapping_time():
    global status
    status.destroy()
    for i in range(6):
        for j in range(4):
            tkimporttubes[i][j].destroy()
            if importtube[i][j][0] == 'b':
                tkimporttubes[i][j] = tk.Button(my_w,text=' ',width=4,bg=colorkey['b'],height=2)
                importtube[i][j] = 'b'
            else:
                tkimporttubes[i][j] = tk.Button(my_w,text=' ',width=4,bg=colorkey['e'],height=2)  
                importtube[i][j] = 'e'
            tkimporttubes[i][j].grid(row=3-j+2,column=i+2)
    important = ()
    lasttube = [[[] for i in range(4)] for i in range(6)]
    lastimage = 0#to compare for motion
    #cv.imshow('frame', img)
    motionamount = 0
    mistake = False
    while True:
        #time.sleep(0.2)
        #print (important)
        ret, img = cam.read()
        img = warp(img)
        #lower = np.array((140,100,60), dtype=np.uint8)
        #upper = np.array((250,160,100), dtype=np.uint8)
        #mask = cv.inRange(img, lower, upper)
        #coloramount = (cv.countNonZero(mask))/mask.shape[0]/mask.shape[1]
        
        grayImage = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        if type(lastimage) != int:
            #(thresh, blackAndWhiteImage) = cv.threshold(grayImage, 127, 255, cv.THRESH_BINARY)

            frameDelta = cv.absdiff(lastimage, grayImage)
            thresh = cv.threshold(frameDelta, 25, 255, cv.THRESH_BINARY)[1]
            motionamount = cv.countNonZero(thresh)#/thresh.shape[0]/thresh.shape[1]
            #print (motionamount)

            #cv.imshow('frame', frameDelta)
            

        lastimage = grayImage[:]
        #key = cv.waitKey(1) & 0xFF
        if cv.waitKey(1) == ord("q"):
            break

        if motionamount > 20:#make sure no one is inside machine
            time.sleep(0.3)
        else:
            indivudualtubes = [[[] for i in range(4)] for i in range(6)]
            colorinfo = [[[] for i in range(4)] for i in range(6)]
            lower = np.array((3,3,3), dtype=np.uint8)
            upper = np.array((90,90,90), dtype=np.uint8)
            for box in boundingboxes:
                x,y,w,h,sy,sx = box
                indivudualtubes[sy][sx]=img[y:y+h, x:x+w]
            for i in range(6):
                for j in range(4):
                    tube = indivudualtubes[i][j]
                    h,w,_ = tube.shape
                    tube = cv.circle(tube, (60,60),80, (0,0,0), 30)
                    blackpix = np.sum(tube == 0)
                    area = h*w
                    average_color_row = np.sum(indivudualtubes[i][j], axis=0)
                    average_color = np.sum(average_color_row, axis=0)
                    biggest = max(average_color)/100
                    B,G,R= int((average_color[0]/(w*h-blackpix))), int((average_color[1]/(w*h-blackpix))), int((average_color[2]/(w*h-blackpix)))
                    Deviation = (max(B,G,R)-min(B,G,R))
                    if (Deviation) < 15:#must be blank or empty
                        low = np.array(3, dtype=np.uint8)
                        upp = np.array(100, dtype=np.uint8)
                        mask = cv.inRange(tube, low, upp)
                        coloramount = (cv.countNonZero(mask))/area#img size is pointless but if ain't broke don't fix it
                        #print (coloramount)
                        if coloramount > 0.30:
                            colorinfo[i][j] ='b'
                        else:
                            colorinfo[i][j] = 'e'
                    else:
                        colorinfo[i][j] = 'c'#c for something else
            #print (tubeinfo,lasttube)
            my_w.update()


            for i in range(6):
                for j in range(4):

                    if colorinfo[i][j] == 'e' and lasttube[i][j] == 'b':#tube put back ini
                        if important == (i,j):
                            print (i,j, 'good tube put back into rack')
                            good()
                            mistake = False
                        else:
                            print (i,j, 'NOT GOOD MISTAKE MADE')
                            mistake = True
                            bad(i,3-j,important[0],3-important[1])
                        buttonereplace(9-j+2,i+2,'e',i,j)
                        my_w.update()
                    elif colorinfo[i][j] == 'b' and lasttube[i][j] == 'e':
                        buttonereplace(9-j+2,i+2,'b',i,j)
                        my_w.update()


            for i in range(6):
                for j in range(4):

                    if colorinfo[i][j] == 'b' and lasttube[i][j] == 'c' and tubeinfo[i][j] !=['b'] and tubeinfo[i][j] !=['e'] :#tube taken out of rack

                        print (i,j,'tube left rack')
                        important = (i,j)
                        #tubeinfo[i][j] = ['b']
                        buttonereplace(9-j+2,i+2,'b',i,j)
                        my_w.update()


        lasttube = colorinfo[:]
        if mistake == False:
            if lasttube == importtube:
                status = tk.Button(my_w,text='Reagents Are\nNow Ready',width=20,bg='green',height = 4) 
                status.grid(row=6,column=0,rowspan = 3)
                break

    #print (status)
def good():
    #status.destroy()
    #print (status.winfo_exists())
    status = tk.Button(my_w,text=f'Good So Far',width=20,bg='green',height = 4) 
    status.grid(row=6,column=0,rowspan = 3)
def bad(wrongx,wrongy,rightx,righty):
    #status.destroy()
    status = tk.Button(my_w,text=f'Lost Track Of Tubes\n Tube {numkey[wrongx]}{letkey[wrongy]} Is Supose\nTo Be At {numkey[rightx]}{letkey[righty]}',width=20,bg='red',height = 4) 
    status.grid(row=6,column=0,rowspan = 3)
def buttonereplace(y,x,bg,yind,xind):
    tkcurrenttubes[yind][xind].destroy()
    tkcurrenttubes[yind][xind] = tk.Button(my_w,width=4,bg=colorkey[bg],height=2)
    tkcurrenttubes[yind][xind].grid(row=y,column=x)
def compare_reagents():
    global status
    if tubeinfo != [] and importtube != []:
        if tubeinfo == importtube:
            status = tk.Button(my_w,text='Matching',width=20,bg='green',height = 4) 
            status.grid(row=6,column=0,rowspan = 3)
            print(' ')
            print('Matching')
            print(' ')
            l1 = tk.Button(my_w,text='Start Uncapping',width=20,anchor='w',command=lambda:uncapping_time(),height=2)#after reagents are a match, continue to uncapping 
            l1.grid(row=4,column=0)
        else:
            status = tk.Button(my_w,text='Not\nMatching',width=20,bg='red',height = 4) 
            status.grid(row=6,column=0,rowspan = 3)
            print(' ')
            print('NOT MATCHING')
            print(tubeinfo)
            print(' ')
def error(message):

    errormessage = tk.Tk()
    errormessage.title('Message')
    errormessage.geometry("200x50")  # Size of the window
    errormessage.iconbitmap(r'Logo.ico')
    errormessage.resizable(False, False)
    mes = tk.Label(errormessage,text = message)
    mes.pack()
    errormessage.mainloop()


def click_event(event, x, y, flags, params):
    global currentpoint,lastpoint
    if event == cv.EVENT_LBUTTONDOWN:
        lastpoint = currentpoint
        currentpoint = (x,y)
        print(x,y)
        

def calibrate():
    #print (cam)
    currentpoint = ()
    if testingmode:
        img = cv.imread("graytest.jpg")
    else:

        t0.join()
        ret, img = cam.read()
        if ret == False:
            error('No Camera Attached')
    img = cv.resize(img, (0,0), fx=0.25, fy=0.25) 
    lastpoint = ()


    cv.namedWindow('Point Coordinates')
    cv.setMouseCallback('Point Coordinates', click_event)
    while True:
        
        cv.imshow('Point Coordinates',img)
        if lastpoint != currentpoint:
            
            if lastpoint != ():
                img=cv.line(img,lastpoint,currentpoint,(255,0,0),2)
            
            
        k = cv.waitKey(1) & 0xFF#esc ke
        if k == 27:
            break

def upload_file():
    global importtube
    global tkimporttubes
    flipped =0
    if tkimporttubes != []:
        for i in range(6):
            for j in range(4):
                tkimporttubes[i][j].destroy()

    else:
        tkimporttubes  = [[0 for i in range(4)]for j in range(6)]
        l0 = tk.Label(my_w,text='Master')
        l0.grid(row=0,column=2,columnspan=6)
        l1 = tk.Label(my_w,text=' ') 
        l1.grid(row=0,column=1)
        for i in range(6):
            l1=  tk.Label(my_w, text=numkey[i])
            l1.grid(row=1,column=2+i)
        for i in range(4):
            l1=  tk.Label(my_w, text=letkey[i])
            l1.grid(row=2+i,column=1)
    importtube = [[[]for i in range(4)]for j in range(6)]
    #filename = filedialog.askopenfilename()
    
    if 0 == 1:#bruh
        error('No File Uploaded')
        #upltext.set('No')
    
    else:
        #try:
        if True:
            df = pd.read_csv('master.csv')
            df = df.values.tolist()


            for i in range(6):
                for j in range(4):
                    if flipped:
                        slot =df[5-i][3-j]
                    else:
                        slot = df[i][j]
                    
                    importtube[i][j].append(slot[0])
                    if len(slot) == 2:
                        importtube[i][j].append(slot[1])
                        tkimporttubes[i][j] = tk.Button(my_w,text=numberkey[importtube[i][j][1]],width=4,bg=colorkey[importtube[i][j][0]],height=2)  
                    else:
                        tkimporttubes[i][j] = tk.Button(my_w,text=numberkey[importtube[i][j][0]],width=4,bg=colorkey[importtube[i][j][0]],height=2)  
                    tkimporttubes[i][j].grid(row=3-j+2,column=i+2)




    
            #upltext.set('Success')
        #except:
        #    error('Error')
        #compare_reagents()
    #print (df)

    
def snapgrid(y,x):#depeciated
    ysnap = 0
    xsnap = 0
    yclose = 10000
    xclose = 10000
    for i in range(len(snapy)):
        if abs(snapy[i]-y) < yclose:
            ysnap = i
            yclose = abs(snapy[i]-y)
    for i in range(len(snapx)):
        if abs(snapx[i]-x) < xclose:
            xsnap = i
            xclose = abs(snapx[i]-x)
    return (ysnap,xsnap)
def rotateImage( image, angle ):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
    return result
def warp(source):#warping to become rectangle
    #source = source[:, :, [2, 1, 0]]#turn bgr to rgb
    #print ((source.shape[1], source.shape[0]), 'hi')
    #cv2_imshow(img1_color)
    try:
    
        source_corners = np.array(roi)
        target_corners = np.array([(10,10),(650,10),(650,910),(10,910)])
        H, _ = cv.findHomography(source_corners, target_corners, params=None)
        source = cv.warpPerspective(source, H, (source.shape[1], source.shape[0]))#warp cuz camera is not completely flat

        source = source[0:920, 0:650]
        
        return source
    except:
        error('Corners Are Not Calibrated')


my_w = tk.Tk()
my_w.title('Reagent Rack Terminal')
my_w.geometry("500x450")  # Size of the window
my_w.iconbitmap(r'Logo.ico') 
#my_w.iconphoto(r"Logo.ico")
b1 = tk.Button(my_w, text='Compare Reagents',width=20,anchor='w',command= lambda:main(),height=2)#very simple UI 
b1.grid(row=0,column=0,rowspan=2) 

#b2 = tk.Button(my_w, text='Upload',width=10,anchor='w',command= lambda:upload_file())#very simple UI 
#b2.grid(row=1,column=0) 
b3 = tk.Button(my_w, text='Calibrate',width=20,anchor='w',command= lambda:calibrate(),height=2)#very simple UI 
b3.grid(row=2,column=0) 
b4 = tk.Button(my_w, text='Exit',width=20,anchor='w',command= lambda:exit(),height=2)#very simple UI  
b4.grid(row=3,column=0) 

upload_file()
#def threadmain():
#    t1 = Thread(target=main)
#    t1.start()
def multimain():
    for i in range(10):
        main()
def main():
    global tubeinfo
    global tkcurrenttubes
    tkcurrenttubes = [[0 for i in range(4)]for i in range(6)]
    kernel = np.ones((2,2),np.uint8)
    #img = cv.imread("test2.jpg")
    if testingmode:
        img = cv.imread("test2.jpg")
    else:
        t0.join()
        ret, img = cam.read()
    #ret 
    #if ret == False:
    #    error('Camera Not Attached')
    if 0 == 1:#depreciated
        pass
    else:
        #cv.imshow('l',img)
        #cv.waitKey(0)
        #img = cv.imread('opentron2.jpg')#temporary for when using camera for testing
        img = warp(img)
        #cv.rectangle(img, (0, 0), (650, 910), (255,255,255), 64)#hard coded max 7:10
        '''
        newimg = cv.bilateralFilter(img, 5, 175, 175)
        #cv.imshow('l',newimg)
        #cv.waitKey(0)
        newimg = cv.convertScaleAbs(newimg, beta = 60)#255-(min(tuple(img[455][325])*3
        #cv.imshow('l',newimg)
        #cv.waitKey(0)
        newimg = cv.Canny(newimg, 19, 200)
        #cv.imshow('l',newimg)
        #cv.waitKey(0)
        contours, hierarchy = cv.findContours(newimg, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        
        boxes = np.zeros((910, 650, 3), dtype = np.uint8)
        imgcont = np.zeros((910, 650, 3), dtype = np.uint8)
        imgbox = imgcont

        contour_bad = []
        for contour in contours:#3700-4800 is circle
            #approx = cv.approxPolyDP(contour,0.01*cv.arcLength(contour,True),True)
            area = cv.contourArea(contour)
            [X, Y, W, H] = cv.boundingRect(contour)
            #print (area,W,H)
            
            imgcont = cv.drawContours(imgcont, contour, -1, (255,0,0),3)
            #if W < 200 and W > 60 and H < 100 and H > 50 and (area > 3000 or area < 100):#big circle
            imgbox = cv.rectangle(imgbox, (X, Y), (X + W, Y + H), (0,255,0), 2)
            if (W > 100 and H > 100):
                #imgbox = cv.rectangle(img, (X, Y), (X + W, Y + H), (0,255,0), 2)
                box=cv.rectangle(boxes, (X, Y), (X + W, Y + H), (0,255,0), 1)

                #cv.putText(img, str(X) + ',' +str(Y) + ',' + str(W),(X,Y), font, 1,(255, 255, 0), 2)
                #print (area)


        #cv.imshow('D', boxes)
        #cv.waitKey(0)

        boxes = cv.cvtColor(boxes, cv.COLOR_BGR2GRAY)
        box = cv.resize(box, (0,0), fx=0.5, fy=0.5) 


        contours, hierarchy = cv.findContours(boxes,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
        boundingboxes = []
        #print (len(contours))

        for contour in contours:
            #print (cv.contourArea(contour))
            if cv.contourArea(contour) > 10000:
                
                bo = cv.boundingRect(contour)        
                bo += snapgrid(bo[1],bo[0])
                boundingboxes.append(bo)

        #if len(boundingboxes) != 24:
        '''
        if 0 == 1:#stupid ass coding
            pass
            '''
            cv.imshow('Objects Detected', cv.resize(newimg, (0,0), fx=0.5, fy=0.5) )
            cv.waitKey(0)
            cv.imshow('Objects Detected', cv.resize(imgcont, (0,0), fx=0.5, fy=0.5) )
            cv.waitKey(0)
            cv.imshow('Objects Detected', cv.resize(imgbox, (0,0), fx=0.5, fy=0.5) )
            cv.waitKey(0)
            cv.imshow('Objects Detected',box)
            cv.waitKey(0)
            error('OCR Failed To Detect Wells')
            '''


        else:
            #boundingboxes = sorted(boundingboxes , key=lambda k: [k[4], k[5]])
            
            #print (boundingboxes)
            indivudualtubes = [[[] for i in range(4)] for i in range(6)]
            #area = [[0 for i in range(4)] for i in range(6)]
            #print (boundingboxes)
            for box in boundingboxes:
                x,y,w,h,sy,sx = box
                #cropped_image = img[y:y+h, x:x+w]
                indivudualtubes[sy][sx]=img[y:y+h, x:x+w]
                #area[sy][sx]=h*w
                #cv.imshow('Objects Detected',cropped_image)
                #cv.waitKey(0)

            tubeinfo = []
            for i in range(6):
                tubeinfo.append([])#
                for j in range(4):
                    tubeinfo[i].append([])#stores alll information

                    tube = indivudualtubes[i][j]

                    h,w,_ = tube.shape
                    
                    #tube = cv.circle(tube, (50,50), 1, (0,0,0), 75)
                    tube = cv.circle(tube, (60,60),80, (0,0,0), 30)
                    blackpix = np.sum(tube == 0)

                    area = h*w

                    average_color_row = np.sum(indivudualtubes[i][j], axis=0)
                    average_color = np.sum(average_color_row, axis=0)
                    biggest = max(average_color)/100
                    B,G,R= int((average_color[0]/(w*h-blackpix))), int((average_color[1]/(w*h-blackpix))), int((average_color[2]/(w*h-blackpix)))
                    #Gray = round((B+G+R)/3,2)
                    #print (B,G,R, Gray, R/G, R/B)#round(R/G,2), round(R/B,2), manualcolor[i][j],Gray,average_color, w*h, blackpix)
                    #print (R/G, R/B, G/B)

                    for bound in colorbound:
                        
                        if R/G >= bound[0][0] and R/G <= bound[1][0]:
                            if R/B >= bound[0][1] and R/B <= bound[1][1]:
                                if G/B >= bound[0][2] and G/B <= bound[1][2]:
                                    if bound[2] == 'w':
                                        #graytube = cv.cvtColor(indivudualtubes[i][j], cv.COLOR_BGR2GRAY)
                                        graytube = cv.cvtColor(indivudualtubes[i][j], cv.COLOR_BGR2GRAY)
                                        lower = np.array(160, dtype=np.uint8)
                                        upper = np.array(255, dtype=np.uint8)

                                        mask = cv.inRange(tube, lower, upper)
                                        #kernel = np.ones((2,2),np.uint8)
                                        #cv.imshow("Canny Edged", mask)
                                        #cv.waitKey(0)
                                        cv.rectangle(mask,(20,20),(w-20,h-20),0,2)
                                        #mask = cv.erode(mask,kernel,iterations = 3)
                                        #cv.imshow("Canny Edged", mask)
                                        #cv.waitKey(0) 
                                        cv.rectangle(mask,(0,0),(w-1,h-1),255,2)
                                        #cv.imshow("Canny Edged", mask)
                                        #cv.waitKey(0) 
                                        cv.floodFill(mask, None, (0,0), 0)
                                        #cv.imshow("Canny Edged", mask)
                                        #cv.waitKey(0) 
                                        coloramount = (cv.countNonZero(mask))/(area-blackpix)#img size is pointless but if ain't broke don't fix it
                                        #avg_color_per_row = np.average(graytube, axis=0)
                                        #avg_color = np.average(avg_color_per_row, axis=0)
        
                                        #print (blankc,emptyc,waterc,whitec, histos.index(max(histos)))
                                        #print (coloramount)
                                        if coloramount > 0.4:
                                            tubeinfo[i][j].append('w')
                                    #if avg_color > 145:#white or hh
                                        #    tubeinfo[i][j].append('w')
                                        #elif avg_color > 135:
                                        #    tubeinfo[i][j].append('h')

                                        else:
                                            lower = np.array(3, dtype=np.uint8)
                                            upper = np.array(100, dtype=np.uint8)
                                            mask = cv.inRange(tube, lower, upper)
                                            coloramount = (cv.countNonZero(mask))/area#img size is pointless but if ain't broke don't fix it
                                            print (round(coloramount,3))
                                            if coloramount > 0.30:
                                                tubeinfo[i][j].append('b')
                                            else:
                                                tubeinfo[i][j].append('e')
                                    else:
                                        tubeinfo[i][j].append(bound[2])
                                        break
            print (tubeinfo)

            #print(img)
            
            for j in range(6):#collumn
                for ii in range(4):#row

                    if tubeinfo[j][ii][0] == 'o' or tubeinfo[j][ii][0] == 'w':# or tubeinfo[j][ii][0] == 'h':#essentially if cap has number or letter on it
                        numbers = whitenumbers  

                        if tubeinfo[j][ii][0] == 'o':
                            numbers = orangenumbers
                        else:
                            numbers = whitenumbers

                        lower = np.array([210,210,210], dtype=np.uint8)#to look at only the numbers no color
                        upper = np.array([255,255,255], dtype=np.uint8)
                        cap = indivudualtubes[j][ii]
                        cap = cap[10:cap.shape[0]-20,     10:cap.shape[1]-20]
                        #cv.waitKey(0)
                        cap = cv.convertScaleAbs(cap, beta = 34)
                        #cv.imshow("Canny Edged", cap)
                        #cv.waitKey(0)
                        #cap = cv.cvtColor(cap, cv.COLOR_BGR2GRAY)
                        h,w,_ = cap.shape
                        #cv.waitKey(0)
                        cap = cv.inRange(cap, lower , upper)
                        #cv.waitKey(0)
                        
                        
                        cv.rectangle(cap,(0,0),(w-1,h-1),0,2)
                        cv.floodFill(cap, None, (0,0), 255)
                    
                        #cap = cv.resize(cap, (0,0), fx=0.8, fy=0.8) 
                        cap = cv.bitwise_not(cap)
                        #kernel = np.ones((1,1),np.uint8)
                        #cap = cv.erode(cap,kernel,iterations = 3)
                        #cv.imshow("Canny Edged", cap)
                        #cv.waitKey(0)
                        #cap = cv.resize(cap, (0,0), fx=0.5, fy=0.5) 
                        #if its not a match then it runs atleast 2 more times until calling quits
                        charlist = ocr(cap,24)
                        if len(charlist) == 0:
                            charlist = ocr(cap,60)
                            if len(charlist) == 0:
                                charlist += '7'#because 7 is so hard to detect make iti the last options                       

                            #print (charlist, tubeinfo[j][ii], len(charlist))
                            #cv.imshow("Canny Edged", test)
                            #cv.waitKey(0)

                        frequency = [0,0,0,0,0,0,0,0,0]#robustness
                        for char in charlist:
                            for i in range(len(numbers)):
                                if char in numbers[i]:
                                    if char in doubleweight:
                                        frequency[i] += 3
                                    else:
                                        frequency[i] += 1
                        charlist = list(charlist)
                        charlist.sort()
                        charlist = ''.join(charlist)
                        final = finalnumbers[frequency.index(max(frequency))]
                        if final != 'h':
                            #if final == importtube[j][ii][1] or jj == 2:#2 IS MAX I-1
                            tubeinfo[j][ii].append(final)
                            print (tubeinfo[j][ii][0], tubeinfo[j][ii][1], charlist)
                                #break
                        else:
                            #if final == importtube[j][ii][0] or jj == 2:#2 IS MAX I-1
                            tubeinfo[j][ii][0] = final
                            print (tubeinfo[j][ii][0], charlist)
                                #break
                                #code by Tony Wu
            l0 = tk.Label(my_w,text='Current')
            l0.grid(row=6,column=2,columnspan=6)
            l1 = tk.Label(my_w,text=' ') 
            l1.grid(row=0,column=6)
            for i in range(6):
                l1=  tk.Label(my_w, text=numkey[i])
                l1.grid(row=7,column=2+i)
            for i in range(4):
                l1=  tk.Label(my_w, text=letkey[i])
                l1.grid(row=8+i,column=1)

            for j in range(6):#collumn
                for ii in range(4):#row
                    if len (tubeinfo[j][ii]) == 2:
                        tkcurrenttubes[j][ii] = tk.Button(my_w,text=numberkey[tubeinfo[j][ii][1]],width=4,bg=colorkey[tubeinfo[j][ii][0]],height=2)#pasting                      

                    else:
                        tkcurrenttubes[j][ii] = tk.Button(my_w,text=numberkey[tubeinfo[j][ii][0]],width=4,bg=colorkey[tubeinfo[j][ii][0]],height=2)#pasting
                    tkcurrenttubes[j][ii].grid(row=9-ii+2,column=j+2) 

            compare_reagents()

            for i in range(6):
                for j in range(4):
                    if len(tubeinfo[i][j]) == 2: 
                        tubeinfo[i][j] = tubeinfo[i][j][0]+str(tubeinfo[i][j][1])
                    else:
                        tubeinfo[i][j] = tubeinfo[i][j][0]
            
            time = datetime.now()
            ftime = (str(time.year)+'_'+str(time.month)+'_'+str(time.day)+'_'+str(time.hour)+'_'+str(time.minute)+'_'+str(time.second))+'.csv'

            with open(ftime, 'w', newline='') as file:
                writer = csv.writer(file)
                for i in range(6):
                    writer.writerow(tubeinfo[i])


            #print (ftime)
            #print(importtube)

my_w.mainloop()
