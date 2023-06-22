# import the opencv module
import cv2
import numpy as np

VIDEO_FILENAME = '20220621_115302.mp4'
CONTOUR_AREA_THRESHOLD = 250

IMG_OUTPUT_WIDTH = 1024
IMG_OUTPUT_HEIGHT = 768

LINE_COLOR = (0,0,255)


def isPointInRect(x, y, x1, y1, x2, y2):
    if (x1 > x2):
        tx = x1
        x1, x2 = x2, tx
    if (y1 > y2):
        ty = y1
        y1, y2 = y2, ty
    if ((x > x1 and x < x2) and (y > y1 and y < y2)):
        return True
    else:
        return False


def selectTargetedLines(video_filename):
    capture = cv2.VideoCapture(video_filename)
    frame_count = 0
    first_line = []
    second_line = []
    while capture.isOpened():
        _, img = capture.read()
        img = cv2.resize(img, (IMG_OUTPUT_WIDTH, IMG_OUTPUT_HEIGHT))
        frame_count += 1
        if frame_count >= 20:
            cv2.putText(img, "Select the first and second lines.", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 1)
            if not first_line:
                roi = cv2.selectROI("Select lines.", img)
                first_line.extend([[roi[0], roi[1]], [roi[0] + roi[2], roi[1] + roi[3]]])
            if not second_line:
                roi = cv2.selectROI("Select lines.", img)
                second_line.extend([[roi[0], roi[1]], [roi[0] + roi[2], roi[1] + roi[3]]])
        if first_line and second_line:
            cv2.destroyAllWindows()
            break
    return first_line, second_line
    
    
if __name__ == "__main__" :
    distance = 2  # 0.6m
    first_line, second_line = selectTargetedLines(VIDEO_FILENAME) #start and end lines

    print(first_line, second_line)

    # capturing video
    capture = cv2.VideoCapture(VIDEO_FILENAME)
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = 0

    car_speed = 0
    car_status = ""
    is_car_entered = False

    while capture.isOpened():
        # to read frame by frame
        _, img_1 = capture.read()
        _, img_2 = capture.read()
        frame_count += 2

        if img_1 is None or img_2 is None: break

        img_1 = cv2.resize(img_1, (IMG_OUTPUT_WIDTH, IMG_OUTPUT_HEIGHT))
        img_2 = cv2.resize(img_2, (IMG_OUTPUT_WIDTH, IMG_OUTPUT_HEIGHT))
            
        # find difference between two frames
        diff = cv2.absdiff(img_1, img_2)

        # to convert the frame to grayscale
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        # apply some blur to smoothen the frame
        diff_blur = cv2.GaussianBlur(diff_gray, (5, 5), 0)

        # to get the binary image (threshold image)
        _, thresh_bin = cv2.threshold(diff_blur, 20, 255, cv2.THRESH_BINARY)



        cv2.rectangle(img_1, (first_line[0][0], first_line[0][1]), (first_line[1][0], first_line[1][1]), LINE_COLOR, 2)   # Draw first line
        cv2.rectangle(img_1, (second_line[0][0], second_line[0][1]), (second_line[1][0], second_line[1][1]), LINE_COLOR, 2)   # Draw second line
        cv2.rectangle(img_1, (first_line[0][0], first_line[0][1]), (second_line[1][0], second_line[1][1]), (214,27,214), 2)   # Draw ROI
        cv2.putText(img_1, "Status: " + car_status, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 1)
        if car_speed > 0:
            cv2.putText(img_1, "Speed: {:.5f} m/s".format(car_speed), (50,75), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 1)
            cv2.putText(img_1, "Speed: {:.5f} km/h".format(car_speed*3.6), (50,100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 1)
            cv2.putText(img_1, "distance: {:.5f} m".format(distance), (50,125), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 1)
            
        # to find contours
        contours, hierarchy = cv2.findContours(thresh_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros(img_1.shape[:2],dtype=np.uint8)
        
        # to draw the bounding box when the motion is detected
        
        if (len(contours) > 0):
            contour = max(contours, key = cv2.contourArea)
            if cv2.contourArea(contour) > CONTOUR_AREA_THRESHOLD:
                x, y, w, h = cv2.boundingRect(contour)
                M = cv2.moments(contour)
                cx = int(M['m10']/M['m00']) #car center x
                cy = int(M['m01']/M['m00']) #car center y
                cv2.rectangle(img_1, (x,y), (x+w,y+h), (0,255,0), 2)
                if not is_car_entered:
                    if isPointInRect(cx, cy, first_line[0][0], first_line[0][1], second_line[1][0], second_line[1][1]):
                        is_car_entered = True
                        enteredTime= 1.0*frame_count/fps #1st cross time
                        car_status = "Car Entered."
                        LINE_COLOR = (0, 255, 0)
                        #print(f"enter time:{enteredTime} s")
                if is_car_entered:
                    if not isPointInRect(cx, cy, first_line[0][0], first_line[0][1], second_line[1][0], second_line[1][1]):
                        is_car_entered = False
                        leftTime= 1.0*frame_count/fps #1st cross time
                        car_status = "Car Left."
                        car_speed = (distance/(leftTime - enteredTime))
                        LINE_COLOR = (0,0,255)
                        print(f"speed:{car_speed} m/s")
                        

        # display the output
        cv2.namedWindow("Detecting and Tracking Motion...", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Detecting and Tracking Motion...", IMG_OUTPUT_WIDTH, IMG_OUTPUT_HEIGHT)
        cv2.imshow("Detecting and Tracking Motion...", img_1) #img_1
        if cv2.waitKey(200) & 0XFF == 27: # esc
            cv2.destroyAllWindows()
            exit()
