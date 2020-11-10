import numpy as np

#1. 데이터
x=np.array([1,2,3,4,5,6,7,8,9,10])
y=np.array([1,2,3,4,5,6,7,8,9,10])
x_pred=np.array([11,12,13])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#2. 모델 구성
model=Sequential()
model.add(Dense(30, input_dim=1)) 
model.add(Dense(50))
model.add(Dense(700))
model.add(Dense(8))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae']) 
model.fit(x, y, epochs=1000, batch_size=1) 


#4. 평가, 예측
loss=model.evaluate(x, y, batch_size=1)
print("loss : ", loss)
#print("acc : ", acc)

y_pred=model.predict(x_pred)
print('예측 결과물 : \n',y_pred)

#실습 : 노드, 레이어 조절로 최적의 예측값 찾기

