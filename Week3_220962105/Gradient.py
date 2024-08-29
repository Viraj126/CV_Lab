import cv2
import cv2 as cv
import numpy as np

img = cv2.imread('flower.jpg',0)

kernal1 =  np.array([[-1,-2,-1],
                     [0,0,0],
                     [1,2,1]],np.float32)

kernal2 =  np.array([[-1,0,1],
                     [-2,0,2],
                     [-1,0,1]],np.float32)

filt = cv.filter2D(img,ddepth=-1,kernel=kernal1)
filt2 = cv.filter2D(img,ddepth=-1,kernel=kernal2)

cv.imshow('OI',img)
cv.imshow('EI',filt)
cv.waitKey(0)
cv.destroyAllWindows()
