import cv2
import numpy as np


def apply_kernel(image, kernel):
    rows, cols = image.shape
    ksize = kernel.shape[0]
    pad = ksize // 2

    padded_image = np.pad(image, ((pad, pad), (pad, pad)), mode='constant', constant_values=0)

    output = np.zeros_like(image, dtype=np.float64)

    for i in range(rows):
        for j in range(cols):
            region = padded_image[i:i + ksize, j:j + ksize]
            output[i, j] = np.sum(region * kernel)

    return output


def box_filter(image, kernel_size):
    box_kernel = np.ones((kernel_size, kernel_size), dtype=np.float64) / (kernel_size * kernel_size)
    return apply_kernel(image, box_kernel)


def gaussian_filter(image, kernel_size, sigma):
    ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (xx ** 2 + yy ** 2) / sigma ** 2)
    kernel /= np.sum(kernel)
    return apply_kernel(image, kernel)


def main():
    image_path = 'flower.jpg'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Error: Image not found.")
        return

    kernel_size = 5
    sigma = 1.0

    box_filtered_image = box_filter(image, kernel_size)
    gaussian_filtered_image = gaussian_filter(image, kernel_size, sigma)

    box_filtered_image = cv2.normalize(box_filtered_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    gaussian_filtered_image = cv2.normalize(gaussian_filtered_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    cv2.imshow('Original Image', image)
    cv2.imshow('Box Filter Output', box_filtered_image)
    cv2.imshow('Gaussian Filter Output', gaussian_filtered_image)

    cv2.imwrite('box_filtered_image.jpg', box_filtered_image)
    cv2.imwrite('gaussian_filtered_image.jpg', gaussian_filtered_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
