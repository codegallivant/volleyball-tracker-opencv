import cv2

image = cv2.imread('frame.jpg')
hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
 
cv2.imshow('Original image',image)
cv2.imshow('hsvframe', hsvImage)
cv2.imwrite("hsvframe.jpg", hsvImage)

cv2.waitKey(0)
cv2.destroyAllWindows()