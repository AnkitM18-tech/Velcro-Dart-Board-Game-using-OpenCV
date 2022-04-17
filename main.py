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
countHit = 0
ballDetectedList = []
hitBallDrawInfoList = []
totalScore = 0

# Determining the corner points
cornerPoints = [[377,52],[944,71],[261,624],[1058,612]]

with open("./polygons","rb") as f:
    polygonsWithScore = pickle.load(f)
# print(len(polygonsWithScore))

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
    # cv2.imshow("Color Image",imgColor)
    return mask

while True:
    frameCounter += 1
    # Resetting the frameCounter when it reaches the total frame count
    if frameCounter == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        frameCounter = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES,0)
        countHit = 0
    # reading the frames
    success,img = cap.read()
    # img = cv2.imread("./img.png")
    # Getting the cropped board
    imgBoard = getBoard(img)
    # Getting the masked darts
    mask = detectColorDarts(imgBoard)
    # Remove Previous Contours
    for x,img in enumerate(ballDetectedList):
        mask = mask - img
        # cv2.imshow(str(x),mask)
    # getting the contours of the darts
    imgContours, conFound = cvzone.findContours(imgBoard,mask,3500)
    # Detecting the hit on the board
    if conFound:
        countHit += 1
        if countHit == 15:
            ballDetectedList.append(mask)
            # print("Hit Detected")
            countHit = 0
            for polyScore in polygonsWithScore:
                center = conFound[0]["center"]
                poly = np.array([polyScore[0]],np.int32)
                inside = cv2.pointPolygonTest(poly,center,False)
                # print(inside)
                if inside == 1:
                    hitBallDrawInfoList.append([conFound[0]["bbox"],conFound[0]["center"],poly])
                    totalScore += polyScore[1]
    # print(totalScore)
    imgBlank = np.zeros((imgContours.shape[0],imgContours.shape[1],3),np.uint8)
    for bbox,center,poly in hitBallDrawInfoList:
        cv2.rectangle(imgContours,bbox,(255,0,255),2)
        cv2.circle(imgContours,center,5,(0,255,0),cv2.FILLED)
        cv2.drawContours(imgBlank,poly,-1,color=(0,255,0),thickness=cv2.FILLED)
    # make the hit zone transparent
    imgBoard = cv2.addWeighted(imgBoard,0.7,imgBlank,0.5,0)
    # Display points
    imgBoard,_ = cvzone.putTextRect(imgBoard,f"Total Points: {totalScore}",(10,40),scale=2, offset = 20)
    # Stacking images
    imgStack = cvzone.stackImages([imgBoard,imgContours],2,1)
    # saving the images for further operations
    # cv2.imwrite("./imgBoard.png",imgBoard)
    # Displaying the frames
    # cv2.imshow("Video Frame",img)
    # cv2.imshow("Cropped Frame",imgBoard)
    # cv2.imshow("Mask Image",mask)
    # cv2.imshow("Contours Image",imgContours)
    cv2.imshow("Stacked Image",imgStack)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break