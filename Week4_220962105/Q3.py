import cv2
import numpy as np
import matplotlib.pyplot as plt

def display_image(title, img, cmap='gray'):
    """Displays an image with a given title using matplotlib."""
    plt.figure(figsize=(8, 6))
    plt.title(title)
    plt.imshow(img, cmap=cmap)
    plt.axis('off')
    plt.show()

def segment_color(image_path, lower_bound, upper_bound):
    """Segments the image based on the specified color range."""
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # Convert image from BGR to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Convert lower and upper bounds to numpy arrays
    lower_bound = np.array(lower_bound, dtype=np.uint8)
    upper_bound = np.array(upper_bound, dtype=np.uint8)

    # Create a mask for the specified HSV range
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

    # Apply the mask to the image
    segmented_image = cv2.bitwise_and(image, image, mask=mask)

    # Display results
    display_image('Original Image', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    display_image('Mask', mask, cmap='gray')
    display_image('Segmented Image', cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))

# Example usage
path = 'cat.png'
lb = [10, 30, 30]  # Lower bound of HSV
ub = [90, 255, 255]  # Upper bound of HSV
segment_color(path, lb, ub)
