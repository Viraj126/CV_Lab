import cv2
import numpy as np
def kmeans_clustering(image, K, max_iter=100):
    height, width, channels = image.shape
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(pixel_values, K, None, criteria, 10, flags)
    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    res2 = res.reshape((image.shape))
    return labels, centers, res2
image_path = 'taj1.jpeg'
image = cv2.imread(image_path, cv2.IMREAD_COLOR)
labels, centers, clustered_image = kmeans_clustering(image, 3)
cv2.imshow('Original Image', image)
cv2.imshow('Clustered Image', clustered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
print('Cluster Centers:')
print(centers)
