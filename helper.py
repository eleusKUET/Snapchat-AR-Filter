import cv2
import numpy as np

def normalizer(image, lower_bound=0, upper_bound=255):
    return cv2.normalize(image, None, lower_bound, upper_bound, cv2.NORM_MINMAX).astype(np.uint8)

def resize(image, dsize):
    return cv2.resize(image, dsize, dst=None, fx=0, fy=0, interpolation=cv2.INTER_LINEAR)

def show(title, image):
    image = image.copy()
    image = resize(image, (600, 600))
    cv2.imshow(title, image)
    cv2.waitKey(0)