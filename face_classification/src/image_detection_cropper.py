import sys

import cv2
from keras.models import load_model
import numpy as np

from utils.inference import detect_faces
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.inference import load_image

# parameters for loading data and images
image_path = "../datasets/POSE/"
detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
gender_model_path = '../trained_models/gender_models/simple_CNN.81-0.96.hdf5'
font = cv2.FONT_HERSHEY_SIMPLEX

# hyper-parameters for bounding boxes shape
gender_offsets = (30, 60)

# loading models
face_detection = load_detection_model(detection_model_path)
gender_classifier = load_model(gender_model_path, compile=False)

# getting input model shapes for inference
gender_target_size = gender_classifier.input_shape[1:3]


folder_names = []
dataset_path = '../datasets/POSE/'
origin_ext = ".tif"
result_ext = ".png"
for i in range(1, 1042):
    name = str(i)
    while len(name) is not 6:
        name = "0" + name
    folder_names.append(name)

# loading images
for names in folder_names :
    index_file_name = "FaceFP_2.txt"

    try:
        lines = open(dataset_path + names + "/" + index_file_name, "r").readlines()
    except FileNotFoundError:
        continue

    for image_name in lines:
        tok = image_name.split()
        image_name = tok[0] + origin_ext
        result_image_name = tok[0] + result_ext

        image_path = dataset_path + names + "/" + image_name

        print(image_path)

        rgb_image = load_image(image_path, grayscale=False)
        gray_image = load_image(image_path, grayscale=True)
        gray_image = np.squeeze(gray_image)
        gray_image = gray_image.astype('uint8')

        faces = detect_faces(face_detection, gray_image)

        for face_coordinates in faces:
            x1, x2, y1, y2 = apply_offsets(face_coordinates, gender_offsets)
            rgb_face = rgb_image[y1:y2, x1:x2]

            gray_face = gray_image[y1:y2, x1:x2]

        bgr_image = cv2.cvtColor(rgb_face, cv2.COLOR_RGB2BGR)
        cv2.imwrite(dataset_path + names + "/" + result_image_name, bgr_image)

