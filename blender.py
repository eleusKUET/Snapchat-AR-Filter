import cv2
import helper
import numpy as np

def blend(image1, image2):
    w, h = image1.shape[0], image1.shape[1]
    filter = helper.resize(image2, (h, w))
    image1 = image1.copy()
    for i in range(w):
        for j in range(h):
            if filter[i, j][3] != 0:
                image1[i, j] = filter[i, j][:3]
    return image1


