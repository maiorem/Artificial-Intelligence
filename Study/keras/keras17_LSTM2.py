
#1. 데이터
import numpy as np 

x=np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]]) #(4,3)
y=np.array([4,5,6,7]) #(4,)

print("x.shape : ", x.shape)
print("y.shape : ", y.shape)

x=x.reshape(x.shape[0], x.shape[1], 1)  
#x=x.reshape(4,3,1)
print("x.shape : ", x.shape)


#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model=Sequential()
#model.add(LSTM(20, activation='relu', input_shape=(3,1)))   
model.add(LSTM(20, activation='relu', input_length=3, input_dim=1))   #input_shape는 다음과 같이 변형이 가능함. 입력하는 데이터의 길이는 3이고 하나씩 자르므로 1차원이다. 
model.add(Dense(30))
model.add(Dense(100))
model.add(Dense(30))
model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1, validation_split=0.3)

x_input=np.array([5,6,7]) #(3,) ->(1,3,1)
x_input=x_input.reshape(1,3,1)

y_predict=model.predict(x_input)

print("y_predict : ", y_predict)

