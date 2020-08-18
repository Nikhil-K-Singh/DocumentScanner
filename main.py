import cv2
import numpy as np


widthImg,heightImg = 640,480

frameWidth = 640
frameHeight = 480


cap = cv2.VideoCapture(0)   # change from 0 to 1 / -1 if an external webcam is used
cap.set(3,frameWidth) # width with id=3
cap.set(4,frameHeight) # height with id=4
cap.set(10,150) # brightness with id=10


def preprocessing(img):
    """
    - convert to grayscale
    - apply blur 
    - detect edges

    """
    imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur=cv2.GaussianBlur(imgGray,(5,5),1)
    imgCanny=cv2.Canny(imgBlur,200,200)

    # use dilation and erosion combination
    # to make the edges appropriate width
    # and overcome shadow conditions

    kernel = np.ones((5,5))
    imgDilated=cv2.dilate(imgCanny,kernel,iterations=2)
    imgThreshold=cv2.erode(imgDilated,kernel,iterations=1)

    return imgThreshold


def getContours(img):
    biggest= np.array([])
    maxArea=0
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area>5000:
            cv2.drawContours(imgContour,cnt,-1,(255,0,0),3)
            peri=cv2.arcLength(cnt,True)
            approx=cv2.cv2.approxPolyDP(cnt,0.02*peri,True) # approximation of corner point
            if area>maxArea and len(approx)==4:
                biggest=approx
                maxArea=area

    return biggest


while True:
    success, img = cap.read()
    img=cv2.resize(img,(widthImg,heightImg))

    imgContour=img.copy()
    imgThreshold = preprocessing(img)

    biggest=getContours(imgThreshold)
    cv2.imshow("Video",imgThreshold)   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break