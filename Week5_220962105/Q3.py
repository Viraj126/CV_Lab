import cv2
import numpy as np
import os

def compute_hog_descriptor(image, win_size=(64, 128)):
    hog = cv2.HOGDescriptor(_winSize=win_size,
                            _blockSize=(16, 16),
                            _blockStride=(8, 8),
                            _cellSize=(8, 8),
                            _nbins=9)
    hog_desc = hog.compute(image)
    return hog_desc

def load_images_from_folder(folder, size=(64, 128)):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.resize(img, size)
            images.append(img)
    return images

positive_images = load_images_from_folder('path/to/positive_images')
negative_images = load_images_from_folder('path/to/negative_images')

pos_hog_descriptors = [compute_hog_descriptor(img) for img in positive_images]
def sliding_window(image, step_size, window_size):
    for y in range(0, image.shape[0] - window_size[1], step_size):
        for x in range(0, image.shape[1] - window_size[0], step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])
def compute_hog_for_window(image, window_size):
    hog = cv2.HOGDescriptor(_winSize=window_size,
                            _blockSize=(16, 16),
                            _blockStride=(8, 8),
                            _cellSize=(8, 8),
                            _nbins=9)
    hog_desc = hog.compute(image)
    return hog_desc

def extract_windows_and_features(image, step_size, window_size):
    windows = []
    features = []
    for (x, y, window) in sliding_window(image, step_size, window_size):
        window_hog = compute_hog_for_window(window, window_size)
        windows.append((x, y, window))
        features.append(window_hog)
    return windows, features
def euclidean_distance(desc1, desc2):
    desc1 = desc1.flatten()
    desc2 = desc2.flatten()
    return np.linalg.norm(desc1 - desc2)

def compare_windows_with_references(window_features, reference_descriptors, threshold):
    detected_windows = []
    for win_feature in window_features:
        for ref_desc in reference_descriptors:
            dist = euclidean_distance(win_feature, ref_desc)
            if dist < threshold:
                detected_windows.append(win_feature)
                break
    return detected_windows
def detect_humans(image, reference_descriptors, threshold):
    window_size = (64, 128)
    step_size = 8
    windows, features = extract_windows_and_features(image, step_size, window_size)
    detected_features = compare_windows_with_references(features, reference_descriptors, threshold)

    for (x, y, window), feature in zip(windows, features):
        if feature in detected_features:
            cv2.rectangle(image, (x, y), (x + window_size[0], y + window_size[1]), (0, 255, 0), 2)

    cv2.imshow('Detected Humans', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
test_image = cv2.imread('path/to/test_image.jpg')
test_image = cv2.resize(test_image, (640, 480))  # Adjust size if necessary

threshold = 0.5  # You might need to tune this threshold
detect_humans(test_image, pos_hog_descriptors, threshold)
