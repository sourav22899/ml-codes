import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D,Dense,BatchNormalization,Activation,Dropout,Flatten
from keras.layers import Conv2DTranspose,UpSampling2D,Reshape,LeakyReLU,ReLU

G = Sequential()
G.add(Dense(7*7*256,input_dim=100))
G.add(Activation('relu'))
G.add(BatchNormalization())
G.add(Reshape((7,7,256)))
G.add(UpSampling2D())
G.add(Conv2DTranspose(128,(5,5),padding='same'))
G.add(Activation('relu'))
G.add(BatchNormalization())
G.add(UpSampling2D())
G.add(Conv2DTranspose(64,(5,5),padding='same'))
G.add(Activation('relu'))
G.add(BatchNormalization())
G.add(Conv2DTranspose(32,(5,5),padding='same'))
G.add(LeakyReLU(alpha = 0.01))
G.add(BatchNormalization())
G.add(Conv2DTranspose(1,(5,5),padding='same'))
G.add(Activation('sigmoid'))
G.summary()

D = Sequential()
D.add(Conv2D(64,(5,5),strides=(2,2),padding = 'same',input_shape=(28,28,1)))
D.add(LeakyReLU(alpha = 0.01))
D.add(BatchNormalization())
D.add(Dropout(0.4))
D.add(Conv2D(128,(5,5),strides=(2,2),padding = 'same'))
D.add(LeakyReLU(alpha = 0.01))
D.add(BatchNormalization())
D.add(Dropout(0.4))
D.add(Conv2D(256,(5,5),strides=(2,2),padding = 'same'))
D.add(LeakyReLU(alpha = 0.01))
D.add(BatchNormalization())
D.add(Dropout(0.4))
D.add(Conv2D(512,(5,5),strides=(1,1),padding = 'same',input_shape=(28,28,1)))
D.add(LeakyReLU(alpha = 0.01))
D.add(BatchNormalization())
D.add(Dropout(0.4))
D.add(Flatten())
D.add(Dense(1))
D.add(Activation('sigmoid'))
D.summary()

gan = Sequential()
gan.add(G)
gan.add(D)
gan.summary()
