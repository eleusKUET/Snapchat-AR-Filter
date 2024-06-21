import cv2

face_cascade = cv2.CascadeClassifier('frontal_face.xml')
eye_cascade = cv2.CascadeClassifier('frontal_eye.xml')
mouth_cascade = cv2.CascadeClassifier('mouth.xml')
nose_cascade = cv2.CascadeClassifier('nose.xml')

def detect_face(image):
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.05, minNeighbors=10, minSize=(100, 100))
    return faces

def detect_eye(image):
    eyes = eye_cascade.detectMultiScale(image, scaleFactor=1.05, minNeighbors=10, minSize=(100, 100))
    return eyes

def detect_mouth(image):
    mouths = mouth_cascade.detectMultiScale(image, scaleFactor=1.5, minNeighbors=11, minSize=(100, 100))
    return mouths

def detect_nose(image):
    noses = nose_cascade.detectMultiScale(image, scaleFactor=1.5, minNeighbors=11, minSize=(50, 50))
    return noses