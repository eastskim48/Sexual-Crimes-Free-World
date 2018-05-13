import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout
from keras.constraints import maxnorm
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import HDF5Matrix,np_utils
from matplotlib import pyplot as plt
from keras.optimizers import SGD
import h5py
import keras

def extGender(hdf5): #extGender
    return np.array(hdf5).transpose()[2]
def norm(hdf5): #normalize inputs from 0-255 and 0.0-1.0
    return np.transpose(np.array(hdf5).astype('float32')/255.0,[0,2,3,1])
def exceptUnknown(X,Y): #except data whose gender is unknown
    i=0
    while(i<len(Y)):
        if Y[i]==0:
            Y = np.delete(Y,(i),axis=0)
            X = np.delete(X,(i),axis=0)
        else:
            i+=1
    return np.transpose(X.astype('float32')/255.0,[0,2,3,1]),np_utils.to_categorical(Y-1,num_classes=2) #onehot

X_train = np.array(HDF5Matrix('train.hdf5', 'crops'))
Y_train = extGender(HDF5Matrix('train.hdf5', 'labels'))
X_val = np.array(HDF5Matrix('val.hdf5', 'crops'))
Y_val = extGender(HDF5Matrix('val.hdf5', 'labels'))
X_test = np.array(HDF5Matrix('test.hdf5', 'crops'))
Y_test = extGender(HDF5Matrix('test.hdf5', 'labels'))
X_train,Y_train = exceptUnknown(X_train,Y_train)
X_val,Y_val = exceptUnknown(X_val,Y_val)
X_test,Y_test = exceptUnknown(X_test,Y_test)

#hdf5 data import

def createCNNmodel(num_classes):
    model = Sequential()
    model.add(Conv2D(96, kernel_size=(8,8),strides=(2,2), padding='valid', input_shape=(128, 192, 3), activation='relu', W_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(3, 3),strides=(2,2),padding='valid'))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, kernel_size=(5,5),strides=(2,2),activation='relu',padding='same', W_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(3, 3),strides=(2,2),padding='valid'))
    model.add(Conv2D(256, kernel_size=(3,3),strides=(1,1),activation='relu',padding='same', W_constraint=maxnorm(3)))
    model.add(Conv2D(256, kernel_size=(3,3),strides=(1,1),activation='relu',padding='same', W_constraint=maxnorm(3)))
    model.add(Conv2D(128, kernel_size=(3,3),strides=(1,1),activation='relu',padding='same', W_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(3, 3),strides=(2,2),padding='valid'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    epochs = 40  # >>> should be 25+
    lrate = 0.01 #learning rate
    decay = lrate/epochs
    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    print(model.summary())
    return model, epochs

# create our CNN model
model, epochs = createCNNmodel(2)
print("CNN Model created.")

 # fit and run our model
tb_hist = keras.callbacks.TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)
model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=epochs, batch_size=64,callbacks=[tb_hist])
# Final evaluation of the model
scores = model.evaluate(X_test, Y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
print("done")

model_json = model.to_json()
with open("model.json", "w") as json_file : 
    json_file.write(model_json)

model.save_weights("model.h5")
print("Saved model to disk")