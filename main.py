import cv2
from cv2 import medianBlur
import cvzone
import numpy as np
from cvzone.ColorModule import ColorFinder
import pickle

# Getting the video
frameCounter = 0
cap = cv2.VideoCapture("./Videos/Video2.mp4")
colorFinder = ColorFinder(False) # put true for debugging
hsvValues = {"hmin":30,"smin":34,"vmin":0,"hmax":41,"smax":255,"vmax":255}

# Determining the corner points
cornerPoints = [[377,52],[944,71],[261,624],[1058,612]]

def getBoard(img):
    width,height = int(400*1.5), int(380*1.5)
    pts1 = np.float32(cornerPoints)
    pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgOut = cv2.warpPerspective(img,matrix,(width,height))
    for x in range(4):
        cv2.circle(img,(cornerPoints[x][0],cornerPoints[x][1]),15,(0,255,0),cv2.FILLED)
    return imgOut

def detectColorDarts(img):
    imgBlur = cv2.GaussianBlur(img,(7,7),2)
    imgColor, mask = colorFinder.update(imgBlur,hsvValues)
    kernel = np.ones((7,7),np.uint8)
    mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
    mask = cv2.medianBlur(mask,9)
    mask = cv2.dilate(mask,kernel,iterations=4)
    kernel = np.ones((9,9),np.uint8)
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
    cv2.imshow("Color Image",imgColor)
    return mask

while True:
    frameCounter += 1
    # Resetting the frameCounter when it reaches the total frame count
    if frameCounter == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        frameCounter = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES,0)
    # reading the frames
    success,img = cap.read()
    # img = cv2.imread("./img.png")
    # Getting the cropped board
    imgBoard = getBoard(img)
    # Getting the masked darts
    mask = detectColorDarts(imgBoard)
    # getting the contours of the darts
    imgContours, conFound = cvzone.findContours(imgBoard,mask,3500)
    # saving the images for further operations
    # cv2.imwrite("./imgBoard.png",imgBoard)
    # Displaying the frames
    cv2.imshow("Video Frame",img)
    # cv2.imshow("Cropped Frame",imgBoard)
    cv2.imshow("Mask Image",mask)
    cv2.imshow("Contours Image",imgContours)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break