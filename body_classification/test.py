# 0. 사용할 패키지 불러오기
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import HDF5Matrix,np_utils
import numpy as np
from numpy import argmax
from keras.models import load_model
from keras.optimizers import SGD

def extGender(hdf5): #extGender
    return np.array(hdf5).transpose()[2]
def exceptUnknown(X,Y): #except data whose gender is unknown
    i=0
    while(i<len(Y)):
        if Y[i]==0:
            Y = np.delete(Y,(i),axis=0)
            X = np.delete(X,(i),axis=0)
        else:
            i+=1
    return np.transpose(X.astype('float32')/255.0,[0,2,3,1]),np_utils.to_categorical(Y-1,num_classes=2) #onehot
 
X_test = np.array(HDF5Matrix('test.hdf5', 'crops'))
Y_test = extGender(HDF5Matrix('test.hdf5', 'labels'))
X_test,Y_test = exceptUnknown(X_test,Y_test)

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

scores = loaded_model.evaluate(X_test, Y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
print("done")
# 3. 모델 사용하기
#yhat = model.predict_classes(xhat)

#for i in range(5):
#   print('True : ' + str(argmax(y_test[xhat_idx[i]])) + ', Predict : ' + str(yhat[i]))