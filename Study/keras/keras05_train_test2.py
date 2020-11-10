import numpy as np

#1. 데이터
x_train=np.array([1,2,3,4,5,6,7,8,9,10]) #훈련시킬 데이터
y_train=np.array([1,2,3,4,5,6,7,8,9,10]) #훈련시킬 데이터
x_test=np.array([11,12,13,14,15]) #평가할 데이터
y_test=np.array([11,12,13,14,15]) #평가할 데이터
x_pred=np.array([16,17,18]) #예측값을 낼 데이터

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#2. 모델 구성
model=Sequential()
model.add(Dense(100, input_dim=1)) #현재 최적 hidden layers : 100, 700, 1000 
model.add(Dense(700)) 
model.add(Dense(1000))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae']) 
model.fit(x_train, y_train, epochs=1000, batch_size=1) 


#4. 평가, 예측
loss=model.evaluate(x_test, y_test, batch_size=1)
print("loss : ", loss)
#print("acc : ", acc)

y_pred=model.predict(x_pred)
print('예측 결과물 : \n',y_pred)



