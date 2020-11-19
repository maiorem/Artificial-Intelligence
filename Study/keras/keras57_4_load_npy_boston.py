import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

#데이터
x_train=np.load('./data/boston_x_train.npy')
x_test=np.load('./data/boston_x_test.npy')
y_train=np.load('./data/boston_y_train.npy')
y_test=np.load('./data/boston_y_test.npy')

scaler=MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)


################### 1. load_model ########################

#3. 컴파일, 훈련
from tensorflow.keras.models import load_model
model1 = load_model('./save/model_boston.h5')
#4. 평가, 예측
result1 = model1.evaluate(x_test, y_test, batch_size=32)



############## 2. load_model ModelCheckPoint #############

from tensorflow.keras.models import load_model

model2 = load_model('./model/boston-192-4.8466.hdf5')
#4. 평가, 예측
result2 = model2.evaluate(x_test, y_test, batch_size=32)


################ 3. load_weights ##################

# 2. 모델
model3 = Sequential()
model3.add(Dense(80, activation='relu', input_shape=(13,)))
model3.add(Dense(150, activation='relu'))
model3.add(Dropout(0.1))
model3.add(Dense(350, activation='relu'))
model3.add(Dropout(0.2))
model3.add(Dense(700, activation='relu'))
model3.add(Dense(1000, activation='relu'))
model3.add(Dropout(0.3))
model3.add(Dense(480, activation='relu'))
model3.add(Dropout(0.4))
model3.add(Dense(280, activation='relu'))
model3.add(Dense(80))
model3.add(Dense(30))
model3.add(Dense(1))

# model.summary()

# 3. 컴파일

model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model3.load_weights('./save/weight_boston.h5')

#4. 평가, 예측

y_predict1=model1.predict(x_test)
y_predict2=model2.predict(x_test)
y_predict3=model3.predict(x_test)


from sklearn.metrics import mean_squared_error 
def RMSE(y_test, y_pred) :
    return np.sqrt(mean_squared_error(y_test, y_pred))


from sklearn.metrics import r2_score 

print("모델 저장 RMSE : ", RMSE(y_test, y_predict1))
print("모델 저장 R2 : ", r2_score(y_test, y_predict1))
print("체크포인트 RMSE : ", RMSE(y_test, y_predict2))
print("체크포인트 R2 : ", r2_score(y_test, y_predict2))
print("가중치 RMSE : ", RMSE(y_test, y_predict3))
print("가중치 R2 : ", r2_score(y_test, y_predict3))

'''
모델 저장 RMSE :  3.8952139522002343   
모델 저장 R2 :  0.8568748180574228     
체크포인트 RMSE :  2.2752654581970573  
체크포인트 R2 :  0.9511665323153876    
가중치 RMSE :  3.8952139522002343      
가중치 R2 :  0.8568748180574228  
'''