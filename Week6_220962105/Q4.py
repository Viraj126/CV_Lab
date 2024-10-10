#Image Stitching
import cv2
import numpy as np
import random

def solution(left_img, right_img):
    key_points1, descriptor1, key_points2, descriptor2 = get_keypoint(left_img, right_img)
    good_matches = match_keypoint(key_points1, key_points2, descriptor1, descriptor2)
    final_H = ransac(good_matches)

    # Prepare the points for the transformation
    rows1, cols1 = right_img.shape[:2]
    rows2, cols2 = left_img.shape[:2]

    points1 = np.float32([[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
    points2 = np.float32([[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(-1, 1, 2)

    # Transform points2 using the homography
    points2_transformed = cv2.perspectiveTransform(points2, final_H)
    list_of_points = np.concatenate((points1, points2_transformed), axis=0)

    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)

    # Translation matrix to shift the image
    H_translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]]).dot(final_H)

    # Warp the left image
    output_img = cv2.warpPerspective(left_img, H_translation, (x_max - x_min, y_max - y_min))
    output_img[(-y_min):rows1 + (-y_min), (-x_min):cols1 + (-x_min)] = right_img
    return output_img

def get_keypoint(left_img, right_img):
    l_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    r_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    # Create SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and descriptors
    key_points1, descriptor1 = sift.detectAndCompute(l_img, None)
    key_points2, descriptor2 = sift.detectAndCompute(r_img, None)

    return key_points1, descriptor1, key_points2, descriptor2

def match_keypoint(key_points1, key_points2, descriptor1, descriptor2):
    # k-Nearest neighbours between each descriptor
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(descriptor1, descriptor2, k=2)

    # Ratio test to get good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            left_pt = key_points1[m.queryIdx].pt
            right_pt = key_points2[m.trainIdx].pt
            good_matches.append([left_pt[0], left_pt[1], right_pt[0], right_pt[1]])
    return good_matches

def homography(points):
    A = []
    for pt in points:
        x, y = pt[0], pt[1]
        X, Y = pt[2], pt[3]
        A.append([x, y, 1, 0, 0, 0, -X * x, -X * y, -X])
        A.append([0, 0, 0, x, y, 1, -Y * x, -Y * y, -Y])

    A = np.array(A)
    _, _, vh = np.linalg.svd(A)
    H = (vh[-1, :].reshape(3, 3))
    return H / H[2, 2]

def ransac(good_pts):
    best_inliers = []
    final_H = None
    t = 5  # Distance threshold

    for _ in range(5000):
        random_pts = random.sample(good_pts, 4)
        H = homography(random_pts)

        inliers = []
        for pt in good_pts:
            p = np.array([pt[0], pt[1], 1]).reshape(3, 1)
            p_1 = np.array([pt[2], pt[3], 1]).reshape(3, 1)
            Hp = np.dot(H, p)
            Hp /= Hp[2]
            dist = np.linalg.norm(p_1 - Hp)

            if dist < t:
                inliers.append(pt)

        if len(inliers) > len(best_inliers):
            best_inliers, final_H = inliers, H

    return final_H

if __name__ == "__main__":
    left_img = cv2.imread('mountain_left.png')
    right_img = cv2.imread('mountain_center.png')
    result_img = solution(left_img, right_img)
    cv2.imshow("left Image",left_img)
    cv2.imshow("right Image",right_img)
    cv2.imshow("Stitched Image",result_img)
    cv2.imwrite("stitched_img.png",result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
