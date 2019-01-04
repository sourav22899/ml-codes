import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Flatten,Dense,Conv2D
from keras.layers.pooling import MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(train_images,train_label),(test_images,test_label) = fashion_mnist.load_data()
# print (test_images.shape)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# print(class_names)
train_images = train_images/255.0
test_images = test_images/255.0
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_label[i]])

# plt.show()
model = Sequential()
model.add(Conv2D(64,kernel_size = 3,activation = 'relu',input_shape = (28,28,1)))
model.add(Conv2D(64,kernel_size = 3,activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Flatten())
model.add(Dense(64))
model.add(Dense(10,activation = 'softmax'))
print(model.summary())

model.compile(optimizer = 'adam',
                loss = 'sparse_categorical_crossentropy',
                metrics = ['accuracy'])
print(train_images.shape)
print(test_images.shape)
train_images = train_images.reshape(60000,28,28,1)
test_images = test_images.reshape(10000,28,28,1)
model.fit(train_images,train_label,validation_data = (test_images,test_label),epochs = 1,batch_size=32)
test_loss,test_acc = model.evaluate(test_images,test_label)
print (test_acc)
