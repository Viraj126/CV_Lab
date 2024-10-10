import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess images
img1 = cv2.imread('mountain_left.png')
img2 = cv2.imread('mountain_center.png')
img1 = cv2.resize(img1, (750, 1200))
img2 = cv2.resize(img2, (750, 1200))

# Convert images to RGB
img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors
keypoints_1, descriptors_1 = sift.detectAndCompute(img1_rgb, None)
keypoints_2, descriptors_2 = sift.detectAndCompute(img2_rgb, None)

# Function to calculate Euclidean distances
def euclidean_distance(desc1, desc2):
    return np.linalg.norm(desc1 - desc2)

# Nearest neighbor matching
matches = []
for i, desc1 in enumerate(descriptors_1):
    best_distance = float('inf')
    best_index = -1
    for j, desc2 in enumerate(descriptors_2):
        distance = euclidean_distance(desc1, desc2)
        if distance < best_distance:
            best_distance = distance
            best_index = j
    matches.append((i, best_index, best_distance))

# Sort matches by distance
matches = sorted(matches, key=lambda x: x[2])

# Visualize the matches
# We'll use the top 150 matches for visualization
num_matches_to_show = 150
img_matches = cv2.drawMatches(img1_rgb, keypoints_1, img2_rgb, keypoints_2,
                               [cv2.DMatch(m[0], m[1], m[2]) for m in matches[:num_matches_to_show]],
                               None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Display the matches
plt.figure(figsize=(15, 10))
plt.imshow(img_matches)
plt.title('Nearest Neighbor Matches')
plt.axis('off')
plt.show()
