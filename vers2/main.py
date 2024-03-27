import cv2 as cv
import numpy as np

def nothing(x):
    pass

vid = cv.VideoCapture("volleyball_match.mp4")
# cv.namedWindow("Tracking")
# cv.createTrackbar("LH", "Tracking", 0, 255, nothing)
# cv.createTrackbar("LS", "Tracking", 0, 255, nothing)
# cv.createTrackbar("LV", "Tracking", 0, 255, nothing)
# cv.createTrackbar("UH", "Tracking", 255, 255, nothing)
# cv.createTrackbar("US", "Tracking", 255, 255, nothing)
# cv.createTrackbar("UV", "Tracking", 255, 255, nothing)
fps = vid.get(cv.CAP_PROP_FPS)

fourcc = cv.VideoWriter_fourcc(*'mp4v') 
video = cv.VideoWriter('volleyball_detected.mp4', fourcc, fps, (1280, 720))

def circular_reshape(n, img, type, iterations = 1):
  # Creates the shape of the kernel
  size = (n, n)
  shape = cv.MORPH_ELLIPSE
  kernel = cv.getStructuringElement(shape, size)
#   Applies the minimum filter with kernel NxN
  if type == "dilate":
    imgResult = cv.dilate(img, kernel, iterations=iterations)
  elif type == "erode":
    imgResult = cv.erode(img, kernel, iterations=iterations)
  return imgResult

bg_subtractor = cv.createBackgroundSubtractorMOG2()

for i in range(0, 500):
    ret, frame = vid.read()
    if ret:
        # hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        # l_h = cv.getTrackbarPos("LH", "Tracking")
        # l_s = cv.getTrackbarPos("LS", "Tracking")
        # l_v = cv.getTrackbarPos("LV", "Tracking")
        # u_h = cv.getTrackbarPos("UH", "Tracking")
        # u_s = cv.getTrackbarPos("US", "Tracking")
        # u_v = cv.getTrackbarPos("UV", "Tracking")
        # l_b = np.array([l_h, l_s, l_v])
        # u_b = np.array([u_h, u_s, u_v])
        # mask = cv.inRange(frame, l_b, u_b)
        # rgb_mask = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
        # frame = cv.blur(frame, (11,11))

        gray_mask = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray_mask = bg_subtractor.apply(gray_mask)
        # kernel = np.ones((10,10),np.uint8)/25

        # blurFrame = cv.filter2D(gray_mask, -1, kernel)
        # blurFrame = cv.minimumBlur(gray_mask, 50)
        # blurFrame = minimum_filter(5, gray_mask)
        # kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        # gray_mask = cv.filter2D(gray_mask, -1, kernel)
        gray_mask = cv.medianBlur(gray_mask, 7)
        gray_mask = circular_reshape(5, gray_mask, "erode", iterations=2)
        gray_mask = circular_reshape(11, gray_mask, "dilate", iterations=1)
        # gray_mask = circular_reshape(3, gray_mask, "erode", iterations=2)
        # gray_mask = circular_reshape(5, gray_mask, "dilate", iterations=1)
        # gray_mask = circular_reshape(1, gray_mask, "dilate", iterations=2)
        # gray_mask = circular_reshape(3, gray_mask, "dilate", iterations=1)

        
        gray_mask = cv.bitwise_not(gray_mask)
        # kernel = np.ones((3, 3), np.uint8) 

        # gray_mask = cv.erode(gray_mask, kernel)
        circles = cv.HoughCircles(gray_mask, cv.HOUGH_GRADIENT, 1.3, 300, param1=40, param2=10, minRadius=2, maxRadius=15)
        # res = cv.bitwise_and(frame, frame, mask = mask)
        # cv.imshow("frame", frame)
        # cv.imshow("mask", mask)
        # cv.imshow("res", res)
        conts, hierarchy = cv.findContours(gray_mask, cv.RETR_EXTERNAL , cv.CHAIN_APPROX_SIMPLE)

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
            rect = cv.rectangle(gray_mask, (x1, y1), (x2, y2), (255,0,0), 2)
            # pad = 20
            # snips.append(img[x1-pad:x2+pad,y1-pad:y2+pad])
            # snip_locations.append(x1-pad,x2+pad,y1-pad,y2+pad) 

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0,:]:
                # draw the outer circle
                cv.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
                # draw the center of the circle
                cv.circle(frame,(i[0],i[1]),2,(0,0,255),3)
        cv.imshow('detected circles', frame)
        cv.imshow('gray mask', gray_mask)
            # cv.imshow('blurred frame', blurFrame)
        video.write(frame)

        key = cv.waitKey(1)
        if key == 27:
            break

#Threshold
#Blur
#Detect circle using hough transform
video.release()
cv.destroyAllWindows()