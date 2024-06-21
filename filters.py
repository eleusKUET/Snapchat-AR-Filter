import cv2
import numpy as np
import blender
import convolution
import detector
import helper

def glass_filter(image, filter_name, debug=True):
    image = image.copy()
    image2 = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector.detect_face(gray)
    glass = cv2.imread('filters/' + filter_name, -1)

    for (fx, fy, fw, fh) in faces:

        roi_color = image[fy:fy+fh, fx:fx+fw]
        roi_color2 = image2[fy:fy+fh, fx:fx+fw]
        roi_gray = gray[fy:fy+fh, fx:fx+fw]
        eyes = detector.detect_eye(roi_gray)
        sz = len(eyes)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color2, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 3)

        for i in range(1, sz, 2):
            (x1, y1, w1, h1) = eyes[i - 1]
            (x2, y2, w2, h2) = eyes[i]
            w1 = max(w1, w2)
            w2 = max(w1, w2)
            h1 = max(h1, h2)
            h2 = max(h1, h2)
            mnx = min(x1, x2)
            mxx = max(x1 + w1, x2 + w2)
            mny = min(y1, y2)
            mxy = max(y1 + h1, y2 + h2)
            # cv2.rectangle(roi_color2, (mnx, mny), (mxx, mxy), (255, 0, 0), 4)
            side = 50
            up = 5
            mnx = max(0, mnx - side)
            mny = max(0, mny - up)
            mxx = min(roi_gray.shape[0] - 1, mxx + side)
            mxy = min(roi_gray.shape[1] - 1, mxy + up)
            to_set = roi_color[mny:mxy, mnx:mxx]
            to_set = blender.blend(to_set, glass)
            roi_color[mny:mxy, mnx:mxx] = to_set
        image[fy:fy+fh, fx:fx+fw] = roi_color
        image2[fy:fy+fh, fx:fx+fw] = roi_color2

    if (debug):
        helper.show('Detected Eyes in the image', image2)
    return image

def nose_filter(image, debug=True):
    image = image.copy()
    image2 = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector.detect_face(gray)
    nose = cv2.imread('filters/nosefilter2.png', -1)

    for (fx, fy, fw, fh) in faces:
        roi_color = image[fy:fy + fh, fx:fx + fw]
        roi_color2 = image2[fy:fy + fh, fx:fx + fw]
        roi_gray = gray[fy:fy + fh, fx:fx + fw]
        noses = detector.detect_nose(roi_gray)

        cv2.rectangle(image2, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 3)

        for (mx, my, mw, mh) in noses:
            cv2.rectangle(roi_color2, (mx, my), (mx + mw, my + mh), (0, 0, 255), 6)

            roi_blend = roi_color[my:my + mh, mx:mx + mw]
            roi_blend = blender.blend(roi_blend, nose)
            roi_color[my:my + mh, mx:mx + mw] = roi_blend
        image[fy:fy + fh, fx:fx + fw] = roi_color
        image2[fy:fy + fh, fx:fx + fw] = roi_color2
    if debug:
        helper.show('Detected noses in the image', image2)
    return image

def dog_filter(image, debug=True):
    image = image.copy()
    image2 = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector.detect_face(gray)
    dog = cv2.imread('filters/dog.png', -1)

    for (fx, fy, fw, fh) in faces:
        roi_color = image[fy:fy + fh, fx:fx + fw]
        roi_color = blender.blend(roi_color, dog)
        cv2.rectangle(image2, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 3)
        image[fy:fy + fh, fx:fx + fw] = roi_color
        break
    if debug:
        helper.show('Detected noses in the image', image2)
    return image

def mustache_filter(image, debug=True):
    image = image.copy()
    image2 = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector.detect_face(gray)
    mus = cv2.imread('filters/mustache.png', -1)

    for (fx, fy, fw, fh) in faces:
        roi_color = image[fy:fy + fh, fx:fx + fw]
        roi_color2 = image2[fy:fy + fh, fx:fx + fw]
        roi_gray = gray[fy:fy + fh, fx:fx + fw]
        mouths = detector.detect_mouth(roi_gray)

        cv2.rectangle(image2, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 3)

        for (mx, my, mw, mh) in mouths:
            mh //= 2
            my -= mh
            cv2.rectangle(roi_color2, (mx, my), (mx + mw, my + mh), (0, 0, 255), 3)
            roi_blend = roi_color[my:my+mh, mx:mx+mw]

            roi_blend = blender.blend(roi_blend, mus)
            roi_color[my:my+mh, mx:mx+mw] = roi_blend
            break
        image[fy:fy + fh, fx:fx + fw] = roi_color
        image2[fy:fy + fh, fx:fx + fw] = roi_color2
    if debug:
        helper.show('Detected mouths in the image', image2)
    return image

def hat_filter(image, debug=True):
    image = image.copy()
    image2 = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector.detect_face(gray)
    hat = cv2.imread('filters/hat.png', -1)

    for (fx, fy, fw, fh) in faces:
        fh //= 2
        fy -= fh
        roi_color = image[fy:fy + fh, fx:fx + fw]
        roi_color2 = image2[fy:fy + fh, fx:fx + fw]
        roi_gray = gray[fy:fy + fh, fx:fx + fw]
        roi_color = blender.blend(roi_color, hat)

        cv2.rectangle(image2, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 3)

        image[fy:fy + fh, fx:fx + fw] = roi_color
        image2[fy:fy + fh, fx:fx + fw] = roi_color2
    if debug:
        helper.show('Detected top head in the image', image2)
    return image

def histogram_equalization(image):
    image = image.copy()
    mx = np.max(image)
    freq = np.zeros(256)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            freq[int(image[i, j])] += 1
    freq = freq / (np.sum(freq))
    for i in range(1, 256):
        freq[i] += freq[i - 1]
    freq = np.round(freq * mx)
    return freq

def histogram_matching(image1, image2):
    image1 = image1.copy()
    image2 = image2.copy()
    eq_image1 = histogram_equalization(image1)
    eq_image2 = histogram_equalization(image2)
    match_image = np.zeros_like(eq_image1)

    for i in range(256):
        dif = 300
        intensity = 0
        for j in range(256):
            if abs(eq_image1[i] - eq_image2[j]) < dif:
                dif = abs(eq_image1[i] - eq_image2[j])
                intensity = j
        match_image[i] = intensity
    return match_image.astype(np.uint8)

def make_image_from_frequency(image, freq):
    image = image.copy()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i, j] = freq[int(image[i, j])]
    return image.astype(np.uint8)

def equalization_filter(image):
    first, second, third = convolution.channel_seperator(image)
    first = make_image_from_frequency(first, histogram_equalization(first))
    second = make_image_from_frequency(second, histogram_equalization(second))
    third = make_image_from_frequency(third, histogram_equalization(third))
    output = convolution.channel_merger(first, second, third)
    output = helper.normalizer(output)
    return output

def matching_filter(image):
    kernel = cv2.imread('train/skin.png', cv2.IMREAD_COLOR)
    first1, second1, third1 = convolution.channel_seperator(image)
    first2, second2, third2 = convolution.channel_seperator(kernel)

    first1 = make_image_from_frequency(first1, histogram_matching(first1, first2))
    second1 = make_image_from_frequency(second1, histogram_matching(second1, second2))
    third1 = make_image_from_frequency(third1, histogram_matching(third1, third2))
    output = convolution.channel_merger(first1, second1, third1)
    output = helper.normalizer(output)
    return output