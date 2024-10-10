import cv2 as cv
import numpy as np

def compute_lbp(img,dtype = np.uint8):
    lbp_img = np.zeros_like(img,dtype=np.uint8)
    r,c = img.shape
    radius = 1
    nb = 8

    for i in range(radius,r - radius):
        for j in range(radius, c - radius):
            centre_pix = img[i,j]
            bin_str = ''

            for k in range(nb):
                theta = (k*2*np.pi)/nb
                dx = int(radius*np.cos(theta))
                dy = int(radius*np.sin(theta))

                nei_pix = img[i+dy,j+dx]
                bin_str += '1' if nei_pix >= centre_pix else '0'
            lbpval = int(bin_str,2)
            lbp_img[i,j] = lbpval
    return lbp_img

def comp_hist(lbp_img):
    hist,_ = np.histogram(lbp_img.ravel(), bins = np.arange(0,256))
    return hist

img = cv.imread('book1.jpeg',0)
lbp_img = compute_lbp(img)
cv.imshow('lbpimg',lbp_img)
hist = comp_hist(lbp_img)
print('Hist',hist)
cv.waitKey(0)
cv.destroyAllWindows()
