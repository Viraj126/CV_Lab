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

# Nearest neighbor matching with second best match tracking
matches = []
for i, desc1 in enumerate(descriptors_1):
    distances = []
    for j, desc2 in enumerate(descriptors_2):
        distance = euclidean_distance(desc1, desc2)
        distances.append((distance, j))

    # Sort distances to find the best and second best matches
    distances.sort(key=lambda x: x[0])
    best_distance, best_index = distances[0]
    second_best_distance = distances[1][0] if len(distances) > 1 else float('inf')

    matches.append((i, best_index, best_distance, second_best_distance))

# Apply the ratio test to filter matches
ratio_threshold = 0.75
good_matches = []
for match in matches:
    if match[2] < ratio_threshold * match[3]:  # compare best distance with second best
        good_matches.append(match)

# Sort good matches by distance
good_matches = sorted(good_matches, key=lambda x: x[2])

# Extract matched keypoint coordinates
pts1 = np.float32([keypoints_1[m[0]].pt for m in good_matches])
pts2 = np.float32([keypoints_2[m[1]].pt for m in good_matches])

# Estimate homography using RANSAC
H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC)

# Draw inlier matches
inlier_matches = [good_matches[i] for i in range(len(mask)) if mask[i]]
img_matches_inliers = cv2.drawMatches(img1_rgb, keypoints_1, img2_rgb, keypoints_2,
                                       [cv2.DMatch(m[0], m[1], 0) for m in inlier_matches],
                                       None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Display the matches
plt.figure(figsize=(15, 10))
plt.imshow(img_matches_inliers)
plt.title('Inlier Matches after Homography Estimation using RANSAC')
plt.axis('off')
plt.show()

# Print the estimated homography matrix
print("Estimated Homography Matrix:")
print(H)
