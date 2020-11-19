#2D를 선으로 쭉 펼친 데이터 모델.
#3차원으로 구성되고 input_shape는 2차원 => LSTM과 동일
#LSTM = 연속된 데이터로 다음 데이터를 찾는 모델
#::Conv1D = 연속된 데이터로 다음 특성을 추출. 이미지와 시계열 둘 다 씀.
#LSTM은 연산량이 많아 속도가 느리므로 먼저 Conv1D로 시도하는 것도 좋은 방법...

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten
from tensorflow.keras.layers import MaxPooling1D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras25_split import split_x 

#1. 데이터
a=np.array(range(1,100))
size=5

#split_x 함수 
datasets=split_x(a, size)

x=datasets[:,:4]
y=datasets[:,4]

x_train, x_test, y_train, y_test=train_test_split(x, y, train_size=0.8)
x_predict=np.array([[97, 98, 99, 100]])

scaler=MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
x_predict=scaler.transform(x_predict)

x_train=x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test=x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
x_predict=x_predict.reshape(x_predict.shape[0], x_predict.shape[1], 1)


#2. 모델 구성 : LSTM ver
model=Sequential()
model.add(LSTM(20, input_shape=(x_train.shape[1], 1)))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(80))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(1))


#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 예측
loss=model.evaluate(x_test, y_test, batch_size=1)

y_predict=model.predict(x_predict)

print("y_predict :", y_predict)
print("loss : ", loss)

# y_predict : [[101.03154]]
# loss :  0.09404636174440384