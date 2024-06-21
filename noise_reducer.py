import cv2
import convolution
import numpy as np

def gaussian_blur(img, sigma, ksize):
    kernel = convolution.gaussian_kernel((ksize, ksize), sigma)
    output = convolution.fast_convolution(img, kernel)
    return convolution.normalizer(output)

def salt_pepper_reduce(image, ksize):
    output = np.zeros_like(image)
    id = (ksize * ksize) // 2
    cx, cy = (ksize + 1) // 2 - 1, (ksize + 1) // 2 - 1

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if i + ksize > image.shape[0] or j + ksize > image.shape[1]:
                continue
            med = []
            for k in range(ksize):
                for l in range(ksize):
                    med.append(image[i + k, j + l])
            med = sorted(med)
            output[i + cx, j + cy] = med[id]
    return output

def reduce(image, sigma, ksize):
    output = salt_pepper_reduce(image, ksize=3)
    output = gaussian_blur(output, sigma, ksize)
    return output
