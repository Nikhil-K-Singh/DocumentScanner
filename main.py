import numpy as np
import cv2

CAP = cv2.VideoCapture(0)   # change from 0 to 1 / -1 if an external webcam is used


def getWarp(img,biggest):
    biggest=reorder(biggest)
    imgWarped=getWarp(img,biggest)

    pts1 = np.float32(biggest)

    pts2 = np.float32([
        [0,0],
        [widthImg,0],
        [0,heightImg],
        [widthImg,heightImg]
    ])

    matrix =cv2.getPerspectiveTransform(pts1,pts2)

    imgOutput = cv2.warpPerspective(img,matrix,(width,height))
    

    ## slight cropping to prevent any imperfections from teh boundary of teh scanned image
    ##
    imgCropped = imgOutput[
        20:imgOutput.shape[0]-20,
        20:imgOutput.shape[1]-20
        ]
    
    return imgCropped


def reorder(myPoints):
    myPoints = myPoints.reshape((4,2))
    myPointsNew=np.zeros((4,1,2),np.int32)

    add = myPoints.sum(axis=1)

    myPointsNew[0]=myPoints[np.argmin(add)]
    myPointsNew[3]=myPoints[np.argmax(add)]

    dif = np.diff(myPoints,axis=1)
    
    myPointsNew[1]=myPoints[np.argmin(dif)]
    myPointsNew[2]=myPoints[np.argmin(dif)]

    return myPointsNew


def getContours(img):
    biggest= np.array([])
    maxArea=0
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area>5000:           ##ref~1
            # cv2.drawContours(imgContour,cnt,-1,(255,0,0),3)
            peri=cv2.arcLength(cnt,True)
            approx=cv2.cv2.approxPolyDP(cnt,0.02*peri,True) # approximation of corner point
            if area>maxArea and len(approx)==4: # make sure that the scanned object this a quadrilateral
                                                # and has the max area (thereby the focus of teh user)
                biggest=approx
                maxArea=area
    cv2.drawContours(imgContour,biggest,-1,(255,0,0),20)
    return biggest



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

# Setting the basic CONSTANTS

widthImg,heightImg = 480,640

frameWidth = 640
frameHeight = 480


#Proceeding with the frame capturing

CAP.set(3,frameWidth) # width with id=3
CAP.set(4,frameHeight) # height with id=4
CAP.set(10,150) # brightness with id=10


while True:
    success, img = CAP.read()
    img=cv2.resize(img,(widthImg,heightImg))

    imgContour=img.copy()
    imgThreshold = preprocessing(img)
    # print('''
    # Presently the script will throw an error in case a document is not available...
    # Need to catch the error as an exception and have a walk around it 
    # ''')

    biggest=getContours(imgThreshold)

    # the default contour area to filter to is set at 5000,
    # to have a smaller {with higer resolution} doc
    # user may have to change the value in getContours()
    # ##ref~1

    if biggest.size !=0:
        imgWarped=getWarp(img,biggest)
        cv2.imshow("Document Detected", imgWarped)    # this will be true only  if contour is large enough ## ref~1
        save_file_name = input('''
        File name to save with .jpg extension 

        share the path if dir is to be changed
        
        ''')
        cv2.imwrite(save_file_name,imgWarped)
        pdf_check=input("do you want a pdf save as well [y/n] :\t")
        if pdf_check.lower() in ["y" or "yes"]:
            from PIL import Image
            im=Image.open(save_file_name+".jpg")
            im =im.convert('RGB')
            im.save(f'{save_file_name}.pdf')            
    else:
        cv2.imshow("Image",img)
        cv2.imshow("Thresholded",imgThreshold)
        # cv2.imshow("Warped",imgWarped) # the warping will be done only if documnet is detected

  
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break