# 0. 사용할 패키지 불러오기

from statistics import mode

import cv2
from cv2 import WINDOW_NORMAL
from keras.models import load_model, Sequential
from keras.utils import np_utils
from keras.layers import Dense, Activation
from keras.utils import HDF5Matrix,np_utils
from keras.optimizers import SGD
from keras.preprocessing.image import load_img
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from utils.inference import load_image
import numpy as np
from numpy import argmax
from matplotlib import pyplot as plt
from PIL import Image
# import vlc

from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input

# parameters for loading data and images
body_model_path = '../trained_models/detection_models/haarcascade_fullbody.xml'
detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
gender_model_path = '../trained_models/gender_models/simple_CNN.81-0.96.hdf5'
video_path = '../videos/test2.mp4'
alert_path ='../sounds/ss'

gender_labels = get_labels('imdb')
font = cv2.FONT_HERSHEY_SIMPLEX
# alert = vlc.mediaPlayer(alert_path)

# hyper-parameters for bounding boxes shape
frame_window = 10
gender_offsets = (30, 60)

# loading models
face_detection = load_detection_model(detection_model_path)
body_detection = cv2.CascadeClassifier(body_model_path)
gender_classifier = load_model(gender_model_path, compile=False)

# getting input model shapes for inference
gender_target_size = gender_classifier.input_shape[1:3]

# starting lists for calculating modes
gender_window = []

#------동영상으로 분류------
vidcap = cv2.VideoCapture(video_path)
count = 0
#vidcap.isOpened() 

# img = Image.fromarray(captured, 'RGB')
#------동영상으로 분류------

# 2. 모델 불러오기
from keras.models import model_from_json
json_file = open("model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

while(count<10000000):
    # read()는 grab()와 retrieve() 두 함수를 한 함수로 불러옴
    # 두 함수를 동시에 불러오는 이유는 프레임이 존재하지 않을 때
    # grab() 함수를 이용하여 return false 혹은 NULL 값을 넘겨 주기 때문
    ret, captured = vidcap.read()
    count+=1;
    if(count%3!=0):
        continue;

    # gray_image = load_image("../images/test.png", grayscale=False)

    gray_image = cv2.cvtColor(captured, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(captured, cv2.COLOR_BGR2RGB)
    faces = detect_faces(face_detection, gray_image)
    print(len(faces))

    for face_coordinates in faces:

        x1, x2, y1, y2 = apply_offsets(face_coordinates, gender_offsets)
        rgb_face = rgb_image[y1:y2, x1:x2]
        try:
            rgb_face = cv2.resize(rgb_face, (gender_target_size))
        except:
            continue
        rgb_face = np.expand_dims(rgb_face, 0)
        rgb_face = preprocess_input(rgb_face, False)
        gender_prediction = gender_classifier.predict(rgb_face)

        gender_label_arg = np.argmax(gender_prediction)
        gender_text = gender_labels[gender_label_arg]
        gender_window.append(gender_text)

        if len(gender_window) > frame_window:
            gender_window.pop(0)
        try:
            gender_mode = mode(gender_window)
        except:
            continue

        if gender_text == gender_labels[0]:
            color = (0, 0, 255)
        else:
            color = (255, 0, 0)

        draw_bounding_box(face_coordinates, rgb_image, color)
        print(gender_mode)
        draw_text(face_coordinates, rgb_image, gender_mode,
                  color, 0, -20, 1, 1)


    gray_image = cv2.cvtColor(captured, cv2.COLOR_BGR2GRAY)
    bodies = body_detection.detectMultiScale(gray_image, 1.1)

    images=[]
    for body in bodies:
        # img = Image.open(img_path)
        img = Image.fromarray(captured)          #사진으로 테스트할 때
        area=(body[0],body[1],body[0]+body[2],body[1]+body[3])
        cropped_img = img.crop(area)
        img = cropped_img.resize((128, 192), Image.ANTIALIAS)
        img=img.transpose(Image.ROTATE_90)
        images.append(img)

    inputs=[]
    for img in images:
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        inputs.append(x)
    # epochs = 50  # >>> should be 25+
    # lrate = 0.01 #learning rate
    # decay = lrate/epochs
    # sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    # loaded_model.compile(loss="binary_crossentropy", optimizer=sgd, metrics=['accuracy'])

    for x in inputs:
        predict = loaded_model.predict(x,verbose=0)[0]
        if(predict[0]>= predict[1]):
            result = "man"
        else:
            result = "woman"
        print("predicted gender : %s" % result)
    # 3. 모델 사용하기

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('window_frame', bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vidcap.release()

# from statistics import mode

# import cv2
# from cv2 import WINDOW_NORMAL
# from keras.models import load_model
# import numpy as np

# from utils.datasets import get_labels
# from utils.inference import detect_faces
# from utils.inference import draw_text
# from utils.inference import draw_bounding_box
# from utils.inference import apply_offsets
# from utils.inference import load_detection_model
# from utils.preprocessor import preprocess_input

# # parameters for loading data and images
# detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
# gender_model_path = '../trained_models/gender_models/simple_CNN.81-0.96.hdf5'
# gender_labels = get_labels('imdb')
# font = cv2.FONT_HERSHEY_SIMPLEX

# # hyper-parameters for bounding boxes shape
# frame_window = 10
# gender_offsets = (30, 60)

# # loading models
# face_detection = load_detection_model(detection_model_path)
# gender_classifier = load_model(gender_model_path, compile=False)

# # getting input model shapes for inference
# gender_target_size = gender_classifier.input_shape[1:3]

# # starting lists for calculating modes
# gender_window = []

# # starting video streaming
# cv2.namedWindow('window_frame', WINDOW_NORMAL)
# video_capture = cv2.VideoCapture(0)
# while True:

#     bgr_image = video_capture.read()[1]
#     gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
#     rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
#     faces = detect_faces(face_detection, gray_image)

#     for face_coordinates in faces:

#         x1, x2, y1, y2 = apply_offsets(face_coordinates, gender_offsets)
#         rgb_face = rgb_image[y1:y2, x1:x2]
#         try:
#             rgb_face = cv2.resize(rgb_face, (gender_target_size))
#         except:
#             continue
#         rgb_face = np.expand_dims(rgb_face, 0)
#         rgb_face = preprocess_input(rgb_face, False)
#         gender_prediction = gender_classifier.predict(rgb_face)

#         gender_label_arg = np.argmax(gender_prediction)
#         gender_text = gender_labels[gender_label_arg]
#         gender_window.append(gender_text)

#         if len(gender_window) > frame_window:
#             gender_window.pop(0)
#         try:
#             gender_mode = mode(gender_window)
#         except:
#             continue

#         if gender_text == gender_labels[0]:
#             color = (0, 0, 255)
#         else:
#             color = (255, 0, 0)

#         draw_bounding_box(face_coordinates, rgb_image, color)
#         draw_text(face_coordinates, rgb_image, gender_mode,
#                   color, 0, -20, 1, 1)
#     bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
#     cv2.imshow('window_frame', bgr_image)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break