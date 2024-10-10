import cv2
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
def comp_grad(img):
    Ix = cv.Sobel(img,cv2.CV_64F,1,0,ksize=3)
    Iy = cv.Sobel(img,cv2.CV_64F,0,1,ksize=3)
    return Ix,Iy

def comp_str_tensor(Ix,Iy,window_size=3):
    Ixx = Ix ** 2
    Ixy = Ix * Iy
    Iyy = Iy ** 2
    kernal = cv2.getGaussianKernel(window_size, -1)
    kernal = kernal * kernal.T
    Sxx = cv.filter2D(Ixx,-1,kernal)
    Sxy = cv.filter2D(Ixy, -1,kernal)
    Syy = cv.filter2D(Iyy, -1, kernal)
    return Sxx,Sxy,Syy

def comp_crnr_resp(Sxx,Sxy,Syy,k  =0.04):
    det = (Sxx * Syy) - (Sxy ** 2)
    trace = Sxx + Syy
    R = det - k*(trace **2)
    return R

def non_max_sup(R,threshold):
    corner_rp = np.zeros_like(R)
    dil = cv.dilate(R,None)
    corner_rp = np.where(R == dil,R,0)
    corner_rp[corner_rp<threshold] = 0
    return corner_rp

def harris_crnr(img,windsize = 3,k = 0.04,thresh = 0.01):
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    img = np.float32(img)/255.0
    ix,Iy = comp_grad(img)
    Sxx,Sxy,Syy = comp_str_tensor(ix,Iy,windsize)
    R = comp_crnr_resp(Sxx, Sxy, Syy, k)
    corners = non_max_sup(R,thresh)
    corner_pts = np.argwhere(corners>0)
    return  corner_pts, corners

def draw_corners(img,corner_pts):
    img = cv.imread(img)
    img_rgb = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    for y,x in corner_pts:
        cv2.circle(img_rgb,(x,y),3,(255,0,0),-1)
    plt.figure(figsize=(10,10))
    plt.imshow(img_rgb)
    plt.title('Detected Corner')
    plt.axis('off')

image_path = 'your_image.jpg'
corners, corner_response = harris_crnr(image_path)
draw_corners(image_path, corners)

