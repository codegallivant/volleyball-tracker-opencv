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


kernel = np.ones((3, 3), np.uint16) 

for i in range(0, 500):
    ret, frame = vid.read()
    if ret:
        gray_mask = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray_mask = bg_subtractor.apply(gray_mask)

        gray_mask = cv.medianBlur(gray_mask, 7)
        gray_mask = circular_reshape(2, gray_mask, "erode", iterations=1)
        gray_mask = circular_reshape(9, gray_mask, "dilate", iterations=1)
        gray_mask = cv.threshold(gray_mask, 1, 255, cv.THRESH_BINARY)[1]
        gray_mask_to_input = cv.bitwise_not(gray_mask)
        circles = cv.HoughCircles(gray_mask_to_input, cv.HOUGH_GRADIENT, 0.3, 300, param1=100, param2=5, minRadius=1, maxRadius=30)

        circled_image = gray_mask.copy()
        
        
        # circled_image = cv.bitwise_not(circled_image)
        circled_image[:,:] = 0
        if circles is not None:
          circles = np.uint16(np.around(circles))
          # end  = min(2, len(circles)-1)     
          end = len(circles) - 1
          for i in circles[0:end]:
            # draw the outer circle
            # cv.circle(circled_image,(i[0],i[1]),i[2],255,2)
            # draw the center of the circle
            cv.circle(circled_image,(i[0],i[1]),2,255,3)
        # circled_image = cv.cvtColor(circled_image, cv.COLOR_BGR2GRAY)
        circled_image = circular_reshape(10, circled_image, type="dilate", iterations = 1)

        l_b = np.array([70, 140, 235])
        u_b = np.array([110, 210, 255])
        mask = cv.inRange(frame, l_b, u_b)
        res = cv.bitwise_and(frame, frame, mask = mask)
        # mask = cv.erode(mask, kernel)  
        mask = circular_reshape(10, mask, type="dilate", iterations = 1)
        # mask = cv.bitwise_not(mask)

        combined_mask = cv.bitwise_and(mask, mask, mask = circled_image)
        combined_mask = circular_reshape(10, circled_image, type="dilate", iterations = 1)
        combined_mask = cv.bitwise_or(circled_image, circled_image, mask = combined_mask)
        combined_mask = circular_reshape(10, mask, type="dilate", iterations = 1)
        
        conts, hierarchy = cv.findContours(combined_mask, cv.RETR_EXTERNAL , cv.CHAIN_APPROX_SIMPLE)
    
        for cnt in conts:
          
            area = cv.contourArea(cnt)
            print(area)
            if 900 < area < 1000:
              continue
            #filter more noise
            x, y, w, h = cv.boundingRect(cnt)
            width = w
            height = h
            x1 = x - int(width/2)  # (x1, y1) = top-left vertex
            y1 = y - int(height/2)
            x2 = x1 + width  # (x2, y2) = bottom-right vertex
            y2 = y1 + height
            rect = cv.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)


        cv.imshow('detected circles', frame)
        cv.imshow('gray mask', gray_mask)
        cv.imshow('mask', mask)
        cv.imshow('combined mask', combined_mask)
        cv.imshow('circled image', circled_image)
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