# r2=회귀지수 / max값은 1
# 선형회귀에서 accuracy 대신 많이 씀

import numpy as np

#1. 데이터
x_train=np.array([1,2,3,4,5,6,7,8,9,10]) #훈련 시킬 데이터
y_train=np.array([1,2,3,4,5,6,7,8,9,10]) #훈련 시킬 데이터
x_test=np.array([11,12,13,14,15]) #평가 할 데이터 : 훈련 데이터에 영향을 미쳐선 안됨
y_test=np.array([11,12,13,14,15]) #평가 할 데이터
x_pred=np.array([16,17,18]) #예측값을 낼 데이터

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#2. 모델 구성
model=Sequential()
model.add(Dense(100, input_dim=1)) 
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

y_predict=model.predict(x_test)
print('예측 결과물 : \n',y_predict)


# RMSE (mse에 제곱근 씌운 값) : 사용자 정의
from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE : ", RMSE(y_test, y_predict))


# R2 : accuracy 대신 회귀모델에서 쓰지만 백프로 신뢰할 수는 없어 RMSE와 함께 씀 
from sklearn.metrics import r2_score 

r2=r2_score(y_test, y_predict)
print("R2 : ", r2)