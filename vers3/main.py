import cv2 as cv
import numpy as np
import time


# img = cv.imread("frame.jpg")
vid = cv.VideoCapture("volleyball_match.mp4")

fps = vid.get(cv.CAP_PROP_FPS)
length = int(vid.get(cv.CAP_PROP_FRAME_COUNT))

fourcc = cv.VideoWriter_fourcc(*'mp4v') 
video = cv.VideoWriter('volleyball_detected.mp4', fourcc, fps, (1280, 720))

cv.namedWindow("Tracking")

def nothing(x):
    pass

cv.createTrackbar("LB", "Tracking", 0, 255, nothing)
cv.createTrackbar("LG", "Tracking", 0, 255, nothing)
cv.createTrackbar("LR", "Tracking", 0, 255, nothing)
cv.createTrackbar("UB", "Tracking", 255, 255, nothing)
cv.createTrackbar("UG", "Tracking", 255, 255, nothing)
cv.createTrackbar("UR", "Tracking", 255, 255, nothing)



# imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# Creating kernel 
kernel = np.ones((3, 3), np.uint8) 
# Using cv2.erode() method  

while True:
    ret, img = vid.read()
    # img = cv.blur(img, (5,5))
    # img = cv.bilateralFilter(img,9,75,75)

    if ret:
        # l_b = cv.getTrackbarPos("LB", "Tracking")
        # l_g = cv.getTrackbarPos("LG", "Tracking")
        # l_r = cv.getTrackbarPos("LR", "Tracking")
        # u_b = cv.getTrackbarPos("UB", "Tracking")
        # u_g = cv.getTrackbarPos("UG", "Tracking")
        # u_r = cv.getTrackbarPos("UR", "Tracking")
        # l_b = np.array([l_b, l_g, l_r])
        # u_b = np.array([u_b, u_g, u_r])
        l_b = np.array([65, 130, 235])
        u_b = np.array([110, 210, 255])
        mask = cv.inRange(img, l_b, u_b)
        res = cv.bitwise_and(img, img, mask = mask)
        mask = cv.erode(mask, kernel)  

        # res = cv.cvtColor(res, cv.COLOR_BGR2GRAY)
        cv.imshow("Mask", mask)
        cv.imshow("Res", res)
        conts, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL , cv.CHAIN_APPROX_SIMPLE)

        snips = list()
        snip_locations = list()
        for cnt in conts:
            area = cv.contourArea(cnt)
            #filter more noise
            x, y, w, h = cv.boundingRect(cnt)
            width = 25
            height = 25
            x1 = x - int(width/2)                   # (x1, y1) = top-left vertex
            y1 = y - int(height/2)
            x2 = x1 + width                   # (x2, y2) = bottom-right vertex
            y2 = y1 + height
            rect = cv.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
            # pad = 20
            # snips.append(img[x1-pad:x2+pad,y1-pad:y2+pad])
            # snip_locations.append(x1-pad,x2+pad,y1-pad,y2+pad) 

        video.write(img)

        cv.imshow("Detection", img)


        # time.sleep(0.02)
    key = cv.waitKey(1)
    if key == 27:
        break

video.release()
cv.destroyAllWindows()
# contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

# To decrease false detections -
# Take snippet
# Blur
# Dilate
# Check hough circle existence