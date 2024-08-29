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


def hough_transform(edges, rho_res=1, theta_res=np.pi / 180, threshold=100):
    """Applies the Hough Transform to detect lines in an edge-detected image."""
    height, width = edges.shape
    diag_len = int(np.sqrt(width**2 + height**2))
    num_rhos = int(2 * diag_len / rho_res)
    num_thetas = int(np.pi / theta_res)
    accumulator = np.zeros((num_rhos, num_thetas), dtype=np.int32)
    thetas = np.arange(0, np.pi, theta_res)
    y_coords, x_coords = np.nonzero(edges)

    for x, y in zip(x_coords, y_coords):
        for theta in thetas:
            rho = x * np.cos(theta) + y * np.sin(theta)
            rho_idx = int(rho + diag_len) // rho_res
            theta_idx = int(theta / theta_res) % num_thetas
            accumulator[rho_idx, theta_idx] += 1

    # Detect lines in the accumulator array
    lines = []
    for rho in range(num_rhos):
        for theta in range(num_thetas):
            if accumulator[rho, theta] > threshold:
                rho_value = (rho - diag_len) * rho_res
                theta_value = theta * theta_res
                lines.append((rho_value, theta_value))

    return accumulator, lines


def draw_detected_lines(img, lines):
    """Draws the detected lines on the image."""
    color_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    height, width = img.shape

    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + width * (-b))
        y1 = int(y0 + height * a)
        x2 = int(x0 - width * (-b))
        y2 = int(y0 - height * a)
        cv2.line(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return color_image


if __name__ == "__main__":
    path = 'Chess.jpg'
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Error: Unable to load image from path {path}")
        exit()

    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blurred_image, 50, 150)
    accumulator, lines = hough_transform(edges, threshold=100)
    line_image = draw_detected_lines(image, lines)
    display_image('Original Image', image)
    display_image('Edges', edges)
    display_image('Accumulator', np.log(accumulator + 1))
    display_image('Detected Lines', line_image)
