import cv2

img = cv2.imread('frame.jpg')
 
# img2 = cv2.blur(img, (6,6))
img2 = cv2.bilateralFilter(img,9,75,75)
cv2.imshow('Original image', img)
cv2.imshow('hsvframe', img2)
cv2.imwrite("blurredframe.jpg", img2)

cv2.waitKey(0)
cv2.destroyAllWindows()
