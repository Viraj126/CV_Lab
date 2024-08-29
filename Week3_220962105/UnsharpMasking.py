import cv2
import numpy as np

img=cv2.imread("flower.jpg")
kernel1=np.ones((10,10),dtype=np.float32)/100
fil=cv2.filter2D(src=img,ddepth=-1,kernel=kernel1)
sharp=img-fil
sharpedimg=cv2.addWeighted(img,1,sharp,-0.00025,0)
cv2.imshow("Original Image",img)
cv2.imshow("Blurred Image",fil)
cv2.imshow("Sharped Image",sharpedimg)

cv2.waitKey(0)
cv2.destroyAllWindows()
