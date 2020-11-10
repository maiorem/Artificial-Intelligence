#1. 데이터
import numpy as np   

#(100,3) 데이터 만들기
x=np.array([range(1, 101), range(311, 411), range(100)])
x=np.transpose(x)
y=np.array([range(101, 201), range(711, 811), range(100)])
y=np.transpose(y)

print(x)
print(x.shape) 


# from sklearn.model_selection import train_test_split 

# x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2)
# x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, test_size=0.2)


# #모델 구성
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense

# model=Sequential()
# model.add(Dense(1000, input_shape=(100,3)))
# model.add(Dense(5000))
# model.add(Dense(500))
# model.add(Dense(3))

# #컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')
# model.fit(x_train, y_train, epochs=100, batch_size=5, validation_data=(x_val, y_val))

# #평가, 예측
# y_pred=model.predict(x_test)
# loss=model.evaluate(x_test, y_test, batch_size=5)

# print('결과 : ', y_pred)
# print('loss : ', loss)

# from sklearn.metrics import mean_squared_error 
# def RMSE(y_test, y_pred) :
#     return np.sqrt(mean_squared_error(y_test, y_pred))
# print('RMSE : ', RMSE(y_test, y_pred))

# from sklearn.metrics import r2_score
# r2=r2_score(y_test, y_pred)
# print('R2 : ', r2)
