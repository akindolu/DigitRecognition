import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import pandas as pd
import random
K.set_image_dim_ordering('th')

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load data
dataset = pd.read_csv("./train.csv")
y = dataset[[0]].values.ravel()
X = dataset.iloc[:,1:].values
X_test = pd.read_csv("./test.csv").values

# split data into train = 0.8*number of dataset, validate = 0.2*number of dataset
nRows = X.shape[0]
nColumns = X.shape[1]
validRatio = 0.025
nValid = int(validRatio*nRows)
nTrain = nRows - nValid

validIndex = random.sample(range(nRows), nValid)
trainIndex = numpy.array(list(set(numpy.arange(0,nRows)) - set(validIndex)))
X_valid = X[validIndex,]
X_train = X[trainIndex,]
y_valid = y[validIndex,]
y_train = y[trainIndex,]

# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_valid = X_valid.reshape(X_valid.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_valid = np_utils.to_categorical(y_valid)
num_classes = y_train.shape[1]

def larger_model():
    # create model
    model = Sequential()
    model.add(Convolution2D(30, 5, 5, border_mode='valid', input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(15, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
    
    # build the model
model = larger_model()
# Fit the model
model.fit(X_train, y_train, validation_data=(X_valid, y_valid), nb_epoch=50, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_valid, y_valid, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

y_test = model.predict_classes(X_test, verbose=0)
numpy.savetxt('mnist-dada.csv', numpy.c_[range(1,len(y_test)+1),y_test], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')