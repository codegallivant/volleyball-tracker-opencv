import cv2 as cv
import numpy as np
import time
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
length = vid.get(cv.CAP_PROP_FRAME_COUNT)
# length = 1000

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

# bg_subtractor = cv.createBackgroundSubtractorMOG2()


kernel = np.ones((3, 3), np.uint16) 
frames = [np.zeros((720,1280,3),dtype=np.uint8),np.zeros((720,1280,3),dtype=np.uint8)]
xdetect = 388
ydetect = 243
xdetect = 39
ydetect = 39
for i in range(0, int(length)):
    # print(i)
    ret, frame = vid.read()
    if(i==0):
      previousframe = frame.copy()
      detected = True
      continue
    if ret:
      # print(frame.shape)
      if detected == True:
        sliced_frame = frame[max(0, ydetect-39):min(720-1, ydetect+39), max(0, xdetect-39):min(1280-1, xdetect+39), :].copy()
        sliced_previous_frame = previousframe[max(0, ydetect-39):min(720-1, ydetect+39), max(0, xdetect-39):min(1280-1, xdetect+39), :].copy()
      # print(sliced_frame.shape)

      if (detected == False) or (0 in sliced_frame.shape):
      # if True:
      #   # sliced_frame = frame[0:175,:,:].copy()
        # sliced_previous_frame = previousframe[0:175,:,:].copy()
        sliced_frame = frame.copy()
        sliced_previous_frame = previousframe.copy()
        sliced_frame[180:365] = 0
        sliced_previous_frame[180:365] = 0
        xdetect = 39
        ydetect = 39
      
      detected = False
      # print(sliced_frame.shape, sliced_previous_frame.shape)
    
      gray_frame = cv.cvtColor(sliced_frame, cv.COLOR_BGR2GRAY)
      blurred_gray_frame = cv.medianBlur(gray_frame, 7)

      previous_gray_frame = cv.cvtColor(sliced_previous_frame, cv.COLOR_BGR2GRAY)
      previous_blurred_gray_frame = cv.medianBlur(previous_gray_frame, 7)

      blurred_sliced_frame = cv.medianBlur(sliced_frame, 7)
      # frame = circular_reshape(10, frame, "erode", iterations = 1)

    
      frames[0] = previous_gray_frame.copy()
      frames[1] = gray_frame.copy()
    
      # print(frames[0].shape, frames[1].shape)
    
      frame_diff = cv.absdiff(frames[1],frames[0])

      # thresholded = diff
      thresholded_diff = cv.threshold(frame_diff, 30, 255, cv.THRESH_BINARY)[1]
      eroded_threshed_diff = thresholded_diff
      eroded_threshed_diff = circular_reshape(2, thresholded_diff, "erode", iterations=3)
      eroded_threshed_diff = circular_reshape(3, eroded_threshed_diff, "dilate", iterations = 1)

      l_b = np.array([60, 140, 230])
      u_b = np.array([120, 220, 255])
      # l_b = np.array([39, 130, 235])
      # u_b = np.array([110, 210, 255])
      colour_thresholded = cv.inRange(sliced_frame, l_b, u_b)
      # mask = cv.erode(mask, kernel)  
      colour_thresholded = circular_reshape(5, colour_thresholded, type="dilate", iterations = 3)
      # colour_thresholded = cv.erode(colour_thresholded, kernel)  

      combined_mask = cv.bitwise_and(eroded_threshed_diff, eroded_threshed_diff, mask = colour_thresholded)
      combined_mask = circular_reshape(10, combined_mask, type="dilate", iterations = 1)
      # combined_mask = colour_thresholded.copy()
      # cv.imshow("combined mask", combined_mask)
      conts, hierarchy = cv.findContours(combined_mask, cv.RETR_EXTERNAL , cv.CHAIN_APPROX_SIMPLE)
      
      for cnt in conts:
        area = cv.contourArea(cnt)
        if area > 0:
          x, y, w, h = cv.boundingRect(cnt)
          xdetect = xdetect-39+x
          ydetect = ydetect-39+y
          width = 40
          height = 40
          x1 = xdetect - int(width/2)  # (x1, y1) = top-left vertex
          y1 = ydetect - int(height/2)
          x2 = x1 + width  # (x2, y2) = bottom-right vertex
          y2 = y1 + height
          rect = cv.rectangle(frame, (x1, y1), (x2, y2), 255, 2)
          # print(x, y)
          detected = True
      

        
        
      cv.imshow("eroded threshed difference", eroded_threshed_diff)
      cv.imshow("colour thresholded", colour_thresholded)
      cv.imshow("combined mask", combined_mask)
      cv.imshow("frame", frame)
      cv.imshow("sliced frame", sliced_frame)

      video.write(frame)
      previousframe = frame.copy()

      print(f"{round((i+1)*100/length, 2)}% complete ({i+1}/{int(length)} frames)", end = '\r')
      key = cv.waitKey(1)
      if key == 27:
        break

#Threshold
#Blur
#Detect circle using hough transform
video.release()
cv.destroyAllWindows()