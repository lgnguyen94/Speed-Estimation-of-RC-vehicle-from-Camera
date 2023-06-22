# import the opencv module
import cv2
import numpy as np
import math

# capturing video
IMG_OUTPUT_WIDTH = 1024
IMG_OUTPUT_HEIGHT = 768
capture = cv2.VideoCapture('20220621_115302.mp4')
fps = capture.get(cv2.CAP_PROP_FPS)
CONTOUR_AREA_THRESHOLD = 250

lastCx = 0
lastCy = 0
distance = 0
frameCnt = 0

while capture.isOpened():
    # to read frame by frame
    _, img_1 = capture.read()
    _, img_2 = capture.read()
    if img_1 is None or img_2 is None: break

    img_1 = cv2.resize(img_1, (IMG_OUTPUT_WIDTH, IMG_OUTPUT_HEIGHT))
    img_2 = cv2.resize(img_2, (IMG_OUTPUT_WIDTH, IMG_OUTPUT_HEIGHT))

    # find difference between two frames
    diff = cv2.absdiff(img_1, img_2)

    # to convert the frame to grayscale
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # apply some blur to smoothen the frame
    diff_blur = cv2.GaussianBlur(diff_gray, (5, 5), 0)

    # to get the binary image
    _, thresh_bin = cv2.threshold(diff_blur, 20, 255, cv2.THRESH_BINARY)

    # to find contours
    contours, hierarchy = cv2.findContours(thresh_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(img_1.shape[:2],dtype=np.uint8)

    
    # to draw the bounding box when the motion is detected
    for contour in contours:
        if cv2.contourArea(contour) > CONTOUR_AREA_THRESHOLD:
            M = cv2.moments(contour)
            currentCx = int(M['m10']/M['m00'])
            currentCy = int(M['m01']/M['m00'])
            if lastCx == 0: lastCx = currentCx
            if lastCy == 0: lastCy = currentCy
            frameCnt += 1
            if frameCnt % 2 == 0:
                distance = math.dist([lastCx, lastCy], [currentCx, currentCy])
                frameCnt = 0
                lastCx = currentCx
                lastCy = currentCy
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(mask, (x,y), (x+w,y+h), (255), 2)
            cv2.putText(mask, "v={:.2f} m/s".format(distance/(2*fps)), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36,255,0), 1)
            
            

    countours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for contour in countours:
        cv2.drawContours(img_1,[contour],0,(0, 255, 0), 2)

    # display the output
    cv2.imshow("Detecting Motion...", img_1)
    if cv2.waitKey(50) & 0XFF == 27: # esc
        exit()
