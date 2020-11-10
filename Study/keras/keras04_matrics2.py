import numpy as np

#1. 데이터
x=np.array([1,2,3,4,5,6,7,8,9,10])
y=np.array([1,2,3,4,5,6,7,8,9,10])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#2. 모델 구성
model=Sequential()
model.add(Dense(30, input_dim=1)) 
model.add(Dense(50))
model.add(Dense(70))
model.add(Dense(8))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae', 'acc']) 
model.fit(x, y, epochs=100) 


#4. 평가, 예측
loss=model.evaluate(x, y) #evaluate : loss가 기본 반환. matrics에서 설정한 값도 덧붙여서 리턴(리스트 형태로)

print("loss : ", loss)
#print("acc : ", acc)

# y_pred=model.predict(x)
# print('예측 결과물 : \n',y_pred)



