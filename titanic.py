import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier

# seed = 3
# np.random.seed(seed)
df = pd.read_csv('/home/sourav/Downloads/train.csv',delimiter=',')
df['Sex_bin'] = np.zeros(df.shape[0])
df['Sex_bin'] = df['Sex_bin'].where(df['Sex'] == "male",1)
df['Embarked_bin'] = np.zeros(df.shape[0])
df['Embarked_bin'] = df['Embarked_bin'].where(df['Embarked'] == "Q",1)
df['Embarked_bin'] = df['Embarked_bin'].where(df['Embarked'] == "S",2)
df = df.drop(['PassengerId','Name','Cabin','Sex','Ticket','Embarked'],axis = 1)
df = df.dropna(subset = ['Age'])
X_data = df.iloc[:,1:].values
y_data = df.iloc[:,0].values
print(df.head())
scl = MinMaxScaler()
X_data = scl.fit_transform(X_data)
X_train,X_test,y_train,y_test = train_test_split(X_data,y_data,test_size = 0.2)
sg = SGDClassifier()
sg.fit(X_train,y_train)
pred = sg.predict(X_test)
accuracy = abs(pred - y_test)
print (sum(accuracy)/len(pred))
# model = Sequential()
# model.add(Dense(4,input_shape = (7,)))
# # model.add(Dense(4))
# model.add(Dense(1,activation = 'softmax'))
# print(model.summary())
# model.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics=['accuracy'])
# model.fit(X_train,y_train,epochs=100,validation_data=(X_test,y_test))
# scores = model.evaluate(X_test,y_test)
# print(scores[1]*100)
# print(X_data)