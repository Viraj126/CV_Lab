# Write a simple program to extract RGB values of a pixel
import cv2

path = "dog.jpeg"
x = 100
y = 100
img = cv2.imread(path)
b, g, r = img[y, x]
print(f"RGB values at pixel ({x}, {y}): ({r}, {g}, {b})")
