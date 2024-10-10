import cv2
import numpy as np

def detect_humans_in_image(image_path):
    """
    Detect humans in the given image using OpenCV's pre-trained HoG descriptor.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to load.")

    # Resize image if necessary
    image = cv2.resize(image, (640, 480))  # Adjust size if necessary

    # Load the pre-trained HoG descriptor with a SVM trained for pedestrian detection
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # Perform human detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
    boxes, weights = hog.detectMultiScale(gray, winStride=(8, 8), padding=(8, 8), scale=1.05)

    # Draw bounding boxes around detected humans
    for (x, y, w, h) in boxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show the image with detected humans
    cv2.imshow('Detected Humans', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Test the function with a single image
detect_humans_in_image('HogSample.png')
