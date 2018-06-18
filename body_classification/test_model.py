# 0. 사용할 패키지 불러오기
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import HDF5Matrix,np_utils
import numpy as np
from numpy import argmax
from keras.models import load_model
from keras.optimizers import SGD
from keras.preprocessing.image import load_img
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import cv2
from matplotlib import pyplot as plt
from PIL import Image

#------동영상으로 분류------
vidcap = cv2.VideoCapture('videos/store.mp4')
count = 0
#vidcap.isOpened() 
while(count<10000000):
    # read()는 grab()와 retrieve() 두 함수를 한 함수로 불러옴
    # 두 함수를 동시에 불러오는 이유는 프레임이 존재하지 않을 때
    # grab() 함수를 이용하여 return false 혹은 NULL 값을 넘겨 주기 때문
    if count==1000000:
        ret, captured = vidcap.read()
    count += 1
vidcap.release()
#img = Image.fromarray(captured, 'RGB')
#------동영상으로 분류------


bodyCascade = cv2.CascadeClassifier('/anaconda3/pkgs/opencv-3.3.1-py27hb620dcb_1/share/OpenCV/haarcascades/haarcascade_fullbody.xml')
img_path = "images/test_6.jpeg"
gray = cv2.cvtColor(captured, cv2.COLOR_BGR2GRAY)
bodies = bodyCascade.detectMultiScale(gray, 1.1)
#human body detection

images=[]
for body in bodies:
    img = Image.open(img_path)
    img = captured          #사진으로 테스트할 때
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

# 2. 모델 불러오기
from keras.models import model_from_json
json_file = open("model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
epochs = 50  # >>> should be 25+
lrate = 0.01 #learning rate
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
loaded_model.compile(loss="binary_crossentropy", optimizer=sgd, metrics=['accuracy'])

for x in inputs:
    predict = loaded_model.predict(x,verbose=0)[0]
    if predict[0]==1.0:
        result="man"
    elif predict[1]==1.0:
        result="woman"
    print("predicted gender : %s" % result)
print("done")
# 3. 모델 사용하기