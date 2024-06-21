import cv2
import cv2 as cv
import numpy as np
import math
import scipy.signal

def convolve(image, kernel, center=None):
    ix, iy = image.shape[0], image.shape[1]
    kx, ky = kernel.shape[0], kernel.shape[1]
    #determining center
    cx = (kx + 1) // 2 - 1
    cy = (ky + 1) // 2 - 1

    if center is not None:
        cx, cy = center #user defined center
    right_padding = ky - cy - 1
    down_padding = kx - cx - 1

    px = ix + cx + down_padding
    py = iy + cy + right_padding
    padded_image = np.zeros((px, py))

    for i in range(ix):
        for j in range(iy):
            padded_image[i + cx, j + cy] = image[i, j]

    output = np.zeros((px, py))
    #convolution
    for i in range(px):
        for j in range(py):
            if i + kx < px and j + ky < py:
                total = 0
                for k in range(kx):
                    for l in range(ky):
                        total += padded_image[i + k, j + l] * kernel[-k - 1, - l - 1]
                output[i + cx, j + cy] = total

    new_image = np.zeros((ix, iy))
    for i in range(cx, cx + ix):
        for j in range(cy, cy + iy):
            new_image[i - cx, j - cy] = output[i, j]

    return new_image

def gauss(x, y, sigma):
    #sigma = (sigma_x, sigma_y)
    pw = -0.5 * (x * x / (sigma[0] ** 2) + y * y / (sigma[1] ** 2))
    return 1 / (2 * math.pi * sigma[0] * sigma[1]) * math.exp(pw)

def gaussian_kernel(dim, sigma, center=None):
    cx = (dim[0] + 1) // 2 - 1
    cy = (dim[1] + 1) // 2 - 1
    sigma = (sigma, sigma)
    if center is not None:
        cx, cy = center
    kernel = np.zeros(dim)
    for x in range(dim[0]):
        for y in range(dim[1]):
            kernel[x, y] = gauss(x - cx, y - cy, sigma)
    return kernel

def channel_seperator(colored_image):
    first = np.zeros((colored_image.shape[0], colored_image.shape[1]))
    second = np.zeros((colored_image.shape[0], colored_image.shape[1]))
    third = np.zeros((colored_image.shape[0], colored_image.shape[1]))
    for i in range(colored_image.shape[0]):
        for j in range(colored_image.shape[1]):
            first[i, j] = colored_image[i, j, 0]
            second[i, j] = colored_image[i, j, 1]
            third[i, j] = colored_image[i, j, 2]
    return [first, second, third]

def channel_merger(first, second, third):
    image = np.zeros((first.shape[0], first.shape[1], 3))
    for i in range(first.shape[0]):
        for j in range(first.shape[1]):
            image[i, j, 0] = first[i, j]
            image[i, j, 1] = second[i, j]
            image[i, j, 2] = third[i, j]
    return image

def convolve3D(image, kernel, center=None):
    channels = channel_seperator(image)
    channels[0] = convolve(channels[0], kernel, center)
    channels[1] = convolve(channels[1], kernel, center)
    channels[2] = convolve(channels[2], kernel, center)
    return channel_merger(channels[0], channels[1], channels[2])

def fast_convolution(img, kernel):
    return scipy.signal.fftconvolve(img, kernel, mode='same')
