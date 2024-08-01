import cv2

img=cv2.imread('flower2.jpg')
print(img.shape)
cropped=img[15:352,147:563]
resized=cv2.resize(img,(360,180),interpolation=cv2.INTER_LINEAR)
cv2.imshow('Cropped',cropped)
cv2.imshow('Orignal',img)
cv2.imshow('Resized',resized)
cv2.waitKey(0)
cv2.destroyAllWindows()