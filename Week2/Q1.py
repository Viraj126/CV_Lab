# Write a program to read an image and perform histogram equalization.

import cv2 as cv

img=cv.imread('flower.jpg')

src=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
final_img=cv.equalizeHist(src)

cv.imshow('Original Image',src)
cv.imshow('Equalised Image',final_img)
cv.waitKey(0)