import cv2

path_image = '../images/20240704112131~1.jpg'

image = cv2.imread(path_image)
cv2.imshow('image', image)
cv2.waitKey(0)
