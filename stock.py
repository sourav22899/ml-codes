import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout
import datetime as dt
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler,StandardScaler

df = pd.read_csv('/home/sourav/Downloads/data.csv',delimiter = ',',\
            usecols = ['Date','Open','High','Low','Close','Adj Close','Volume'])

# df['Date'] = pd.to_datetime(df['Date'],dayfirst=True)
df = df.dropna()
"""
plt.figure(figsize=(20,10))
plt.plot(range(df.shape[0]),(df['High']+df['Low'])/2.0)
plt.xticks(range(0,df.shape[0],100),df['Date'].loc[::100],rotation = 45)
plt.xlabel('Date')
plt.ylabel('Midvalue')
plt.show()
"""
# print(df)

# LOOK_BACK = [5,10,15,20,25,30,35,40,45,50,55,60]
# LOOK_BACK = [65,70,75,80,85,90,95,100,105,110,115,120]
# LOOK_BACK = [1,60]
cl = df['Open']
scl = MinMaxScaler(feature_range=(0,0.4))
# scl = StandardScaler()
cl = np.array(cl)
cl = cl.reshape(cl.shape[0],1)
cl = scl.fit_transform(cl)
# print(df)
def processData(data,look_back,look_forward):
    X,Y = [],[]
    for i in range(len(data)-look_back-look_forward+1):
        X.append(data[i:i+look_back,0])
        Y.append(data[i+look_back:i+look_back+look_forward,0])
    # X.append(data[i+look_back:,0])
    # Y.append(data[i+look_back:,0])
    return np.array(X),np.array(Y)

look_backs = 1
split_frac = 0.97
look_forward = 1
X,y = processData(cl,look_backs,look_forward)
X_train,X_test = X[:int(len(X)*split_frac)],X[int(len(X)*split_frac):]
y_train,y_test = y[:int(len(X)*split_frac)],y[int(len(X)*split_frac):]
print(len(X_test),len(y_train))

model = Sequential()
model.add(LSTM(256,return_sequences = False,input_shape = (look_backs,1)))
# model.add(LSTM(256))
model.add(Dropout(0.4))
model.add(Dense(look_forward))
model.compile(optimizer = 'adam',loss = 'mean_squared_error',metrics=['accuracy'])
print(model.summary())
model.save('open_lstm.h5')

X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],1))
X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],1))
# predict the model.In case of volume use high no of epochs ~500-800 to achieve the shape of the graph.
history = model.fit(X_train,y_train,epochs=20,validation_data=(X_test,y_test),shuffle=False)
Xt = model.predict(X_test)
score = model.evaluate(X_test,y_test,verbose=1)
# print(Xt)

y_test = scl.inverse_transform(y_test.reshape(-1,look_forward)) 
Xt = scl.inverse_transform(Xt.reshape(-1,look_forward))
plt.figure(figsize=(18,9))
plt.subplot(1,2,1)
plt.plot(y_test,color = 'red',label = 'actual')
plt.plot(Xt,color = 'blue',label = 'predict')
plt.legend()
error = abs((y_test - Xt)/y_test)
for i in range(len(Xt)):
    print (y_test[i],Xt[i],error[i]*100)
    # error[i] = (y_test[i]-Xt[i])**2
print(y_test.shape)
print(Xt.shape)
print(model.metrics_names)
print(score)
# print(look_backs,100*np.sqrt(np.mean(error**2)))
plt.subplot(1,2,2)
plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
plt.show()

# class DataGenerator(object):
#     def __init__(self,prices,batch_size,num_unroll):
#         self._prices = prices
#         self._prices_length = len(self._prices) - num_unroll
#         self._batch_size = batch_size
#         self._num_unroll = num_unroll
#         self._segments = self._prices_length // self._batch_size
#         self._cursor = [i*self._segments for i in range(self._batch_size)]
    
#     def next_batch(self):
#         batch_data = np.zeros((self._batch_size))
#         batch_labels = np.zeros((self._batch_size))

#         for b in range(self._batch_size):
#             if self._cursor[b]+1 >= self._prices_length:
#                 self._cursor[b] = np.random.randint(0,(b+1)*self._segments)
#             batch_data[b] = self._prices[self._cursor[b]]
#             batch_labels[b] = self._prices[self._cursor[b] + np.random.randint(0,5)]

#             self._cursor[b] = (self._cursor[b]+1)%self._prices_length

#         return batch_data,batch_labels

#     def unroll_batches(self):
#         unroll_data,unroll_labels = [],[]
#         init_data,init_labels = None,None
#         for i in range(self._num_unroll):
#             data,labels = self.next_batch()
#             unroll_data.append(data)
#             unroll_labels.append(labels)
        
#         return unroll_data,unroll_labels
