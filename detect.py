import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D,Dense,Flatten,Dropout,BatchNormalization,Activation
from keras.layers.pooling import MaxPooling2D,AveragePooling2D
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.constraints import maxnorm
from keras.callbacks import TensorBoard,EarlyStopping,ModelCheckpoint

(X_train,y_train),(X_test,y_test) = cifar10.load_data()
seed = 3
np.random.seed(seed)
X_train = X_train/255.0
X_test = X_test/255.0
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

model = Sequential()
model.add(Conv2D(64,(3,3),input_shape = (32,32,3),padding = 'same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3),input_shape = (32,32,3),padding = 'same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Conv2D(128,(3,3),padding = 'same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(128,(3,3),padding = 'same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Conv2D(256,(3,3),padding = 'same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(256,(3,3),padding = 'same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Conv2D(512,(3,3),padding = 'same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(512,(3,3),padding = 'same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(optimizer = 'adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])
print(model.summary())
# tensor_board = TensorBoard(log_dir='./Graph',histogram_freq=0,write_graph=True,write_images=True)
es = EarlyStopping(monitor='val_loss',patience=20,verbose=1,mode='auto')
mc = ModelCheckpoint(monitor='val_loss', filepath='wt_best.hdf5', verbose=1, save_best_only=True)
callback = [es,mc]

import tensorflow as tf

with tf.device('/gpu:0'):
    model.fit(X_train,y_train,batch_size = 32,epochs = 500,validation_split=0.1,callbacks=callback)
scores = model.evaluate(X_test,y_test)
print(scores[1]*100)