import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical


# 1. 데이터
dataset=pd.read_csv('./data/csv/winequality-white.csv', header=0, index_col=None, sep=';')
dataset=dataset.to_numpy()
x=dataset[:,:-1]
y=dataset[:,-1]
print(x.shape) #(4898, 11)
print(y.shape) #(4898,)

x_train, x_test, y_train, y_test=train_test_split(x, y, train_size=0.8)

scale=StandardScaler()
scale.fit(x_train)
x_train=scale.transform(x_train)
x_test=scale.transform(x_test)


y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

print(y_train.shape)
print(y_test.shape)

# 2. 모델
# model=LinearSVC()
# model=SVC()
# model=KNeighborsClassifier()
# model=KNeighborsRegressor()
# model=RandomForestClassifier()
# model=RandomForestRegressor()
model=Sequential()
model.add(Dense(100, activation='relu', input_shape=(11,)))
model.add(Dense(1000, activation='relu'))
model.add(Dense(800, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 3. 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=100, batch_size=10, validation_split=0.2)

# 4. 평가, 예측
# score=model.score(x_test, y_test)

loss, acc=model.evaluate(x_test, y_test, batch_size=10)

y_predict=model.predict(x_test)
y_predict=np.argmax(y_predict, axis=1)
y_actually=np.argmax(y_test, axis=1)

print('loss :', loss)
print('acc :', acc)

print(y_actually[:10], '의 예측 결과 ', y_predict[:10])

'''
loss : 3.515895128250122
acc : 0.6051020622253418
[5 6 5 7 5 7 7 6 7 5] 의 예측 결과  [5 6 5 6 5 7 7 6 7 5]
'''

# 이상치