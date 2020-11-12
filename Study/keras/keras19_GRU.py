
#RNN 기법 중 하나인 GRU 코드

#1. 데이터
import numpy as np 

x=np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]]) #(4,3)
y=np.array([4,5,6,7]) #(4,)

print("x.shape : ", x.shape)
print("y.shape : ", y.shape)

x=x.reshape(x.shape[0], x.shape[1], 1) 
print("x.shape : ", x.shape)


#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU

model=Sequential()
model.add(GRU(10, activation='relu', input_length=3, input_dim=1))   
model.add(Dense(30))
model.add(Dense(100))
model.add(Dense(30))
model.add(Dense(1))

model.summary()

'''
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1, validation_split=0.3)

x_input=np.array([5,6,7]) #(3,) ->(1,3,1)
x_input=x_input.reshape(1,3,1)


# loss=model.evaluate(x_test, y_test, batch_size=1)

y_predict=model.predict(x_input)

# print("loss : ", loss)
print("y_predict : ", y_predict)
'''
