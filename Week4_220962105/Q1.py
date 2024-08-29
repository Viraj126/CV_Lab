import cv2
import numpy as np
import matplotlib.pyplot as plt


def display_image(title, image, cmap='gray'):
    """Displays an image with a given title using matplotlib."""
    plt.figure(figsize=(8, 6))
    plt.title(title)
    plt.imshow(image, cmap=cmap)
    plt.axis('off')
    plt.show()


def binary_inverse_threshold(image, threshold):
    """Applies Binary Inverse Thresholding to a grayscale image."""
    # Ensure the image is in grayscale format (2D array)
    if len(image.shape) != 2:
        raise ValueError("The input image should be a grayscale image.")

    # Create an output image initialized to zeros (black)
    output = np.zeros_like(image)

    # Apply Binary Inverse Thresholding
    output[image < threshold] = 255  # Set pixels below threshold to white
    output[image >= threshold] = 0   # Set pixels above or equal to threshold to black

    return output


path = 'taj2.jpeg'
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
threshold_value = 127
binary_inv_image = binary_inverse_threshold(img, threshold_value)
display_image('Original Image', img)
display_image('Binary Inverse Thresholding', binary_inv_image)

