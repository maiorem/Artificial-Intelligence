#함수형 모델

#1. 데이터
import numpy as np   

#(100,3) 데이터 만들기
x=np.array([range(1, 101), range(311, 411), range(100)])
x=np.transpose(x)
#x=x.transpose()
#x=x.T
y=np.array(range(101, 201))

#print(y.shape)

# print(x)
# print(x.shape) #(100,3)

# y1, y2, y3 = w1x1 + w2x2 + w3x3 + b

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2)
x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, test_size=0.2)


#2. 모델 구성
from tensorflow.keras.models import Sequential, Model 
from tensorflow.keras.layers import Dense, Input

# Sequential 모델
model=Sequential()
model.add(Dense(5, input_shape=(3,), activation='relu')) 
# 활성화함수(activation) 디폴트는 linear
model.add(Dense(4, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(1))

# # 함수형 모델
# input1=Input(shape=(3,))
# dense1=Dense(5, activation='relu')(input1)
# dense2=Dense(4, activation='relu')(dense1)
# dense3=Dense(3, activation='relu')(dense2)
# output1=Dense(1)(dense3) #선형회귀이기 때문에 마지막은 linear여야 함
# model=Model(inputs=input1, outputs=output1)

model.summary()
## param의 값 = 노드 수 * input의 차원 + 노드 수

'''
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=5, validation_data=(x_val, y_val), verbose=3)

#4. 평가, 예측
y_pred=model.predict(x_test)
loss=model.evaluate(x_test, y_test, batch_size=5)

print('결과 : ', y_pred)
print('loss : ', loss)

from sklearn.metrics import mean_squared_error 
def RMSE(y_test, y_pred) :
    return np.sqrt(mean_squared_error(y_test, y_pred))
print('RMSE : ', RMSE(y_test, y_pred))

from sklearn.metrics import r2_score
r2=r2_score(y_test, y_pred)
print('R2 : ', r2)
'''