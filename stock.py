import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout,GRU,SimpleRNN
import datetime as dt
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler,StandardScaler

df = pd.read_csv('/home/sourav/Downloads/data.csv',delimiter = ',',\
            usecols = ['Date','Open','High','Low','Close','Adj Close','Volume'])

# df['Date'] = pd.to_datetime(df['Date'],dayfirst=True)
df = df.dropna()

plt.figure(figsize=(20,10))
plt.plot(range(df.shape[0]),(df['High']+df['Low'])/2.0)
plt.xticks(range(0,df.shape[0],100),df['Date'].loc[::100],rotation = 45)
plt.xlabel('Date')
plt.ylabel('Midvalue')
plt.show()

cl = df['High']
scl = MinMaxScaler(feature_range=(0,1))
cl = np.array(cl)
cl = cl.reshape(cl.shape[0],1)
cl = scl.fit_transform(cl)

def processData(data,look_back,look_forward):
    X,Y = [],[]
    for i in range(len(data)-look_back-look_forward+1):
        X.append(data[i:i+look_back,0])
        Y.append(data[i+look_back:i+look_back+look_forward,0])
    return np.array(X),np.array(Y)

look_backs = 1
split_frac = 0.985
look_forward = 1
X,y = processData(cl,look_backs,look_forward)
X_train,X_test = X[:int(len(X)*split_frac)],X[int(len(X)*split_frac):]
y_train,y_test = y[:int(len(X)*split_frac)],y[int(len(X)*split_frac):]
print('Size of X_train:{0}'.format(len(X_train)))
print('Size of X_test:{0}'.format(len(X_test)))

model = Sequential()
model.add(LSTM(256,return_sequences = False,input_shape = (look_backs,1)))
model.add(Dropout(0.2))
model.add(Dense(look_forward))
model.compile(optimizer = 'adam',loss = 'mean_squared_error',metrics=['accuracy'])
print(model.summary())

X_final = np.zeros_like(X_test)
print(X_final.shape)
X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],1))
X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],1))
# predict the model.In case of volume use high no of epochs ~500-800 to achieve the shape of the graph.
no_of_models = 5
for k in range(no_of_models):
    history = model.fit(X_train,y_train,epochs=20,validation_split=0.015,shuffle=False)
    Xt = model.predict(X_test)
    score = model.evaluate(X_test,y_test,verbose=1)
    Xt = scl.inverse_transform(Xt.reshape(-1,look_forward))
    # print(Xt.shape)
    # print(Xt)
    X_final += Xt

Xt = X_final/no_of_models
y_test = scl.inverse_transform(y_test.reshape(-1,look_forward)) 
model.save('open_lstm.h5')
plt.figure(figsize=(18,9))
plt.subplot(1,2,1)
plt.plot(y_test,color = 'red',label = 'actual')
plt.plot(Xt,color = 'blue',label = 'predict')
plt.legend()

error = abs((y_test - Xt)/y_test)
for i in range(len(Xt)):
    print (i+1,'.', y_test[i],Xt[i],error[i]*100)

print('Mean Error:{0:.5f}'.format(np.mean(error)*100))
print('Shape of X_test:{0}'.format(Xt.shape))
print('Shape of y_test:{0}'.format(y_test.shape))
plt.subplot(1,2,2)
plt.plot(history.history['loss'],color = 'red',label = 'loss')
plt.plot(history.history['val_loss'],color = 'blue',label = 'val_loss')
plt.legend()
plt.show()
