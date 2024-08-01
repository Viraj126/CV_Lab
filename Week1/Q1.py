# 1. Write a simple program to read, display, and write an image.
import cv2

dog = cv2.imread('images.jpeg')
cv2.imshow('dog', dog)
cv2.imwrite('dog.jpeg', dog)
cv2.waitKey(0)
cv2.destroyAllWindows()
