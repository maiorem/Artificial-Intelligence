import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

#데이터 불러오기
x_train=np.load('./data/diabetes_x_train.npy')
x_test=np.load('./data/diabetes_x_test.npy')
y_train=np.load('./data/diabetes_y_train.npy')
y_test=np.load('./data/diabetes_y_test.npy')


scaler=StandardScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)


################### 1. load_model ########################

#3. 컴파일, 훈련
from tensorflow.keras.models import load_model
model1 = load_model('./save/model_diabetes.h5')
#4. 평가, 예측
result1 = model1.evaluate(x_test, y_test, batch_size=32)



############## 2. load_model ModelCheckPoint #############

from tensorflow.keras.models import load_model

model2 = load_model('./model/diabetes-33-2752.4292.hdf5')
#4. 평가, 예측
result2 = model2.evaluate(x_test, y_test, batch_size=32)


################ 3. load_weights ##################

# 2. 모델
model3 = Sequential()
model3.add(Dense(128, activation='relu', input_shape=(10,)))
model3.add(Dense(256, activation='relu'))
model3.add(Dropout(0.2))
model3.add(Dense(64, activation='relu'))
model3.add(Dense(32, activation='relu'))
model3.add(Dense(8, activation='relu'))
model3.add(Dense(1))

# model.summary()

# 3. 컴파일

model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model3.load_weights('./save/weight_diabetes.h5')

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
모델 저장 RMSE :  43.59025457855183    
모델 저장 R2 :  0.6728169219228686     
체크포인트 RMSE :  45.951944963943845  
체크포인트 R2 :  0.6364033952309502    
가중치 RMSE :  43.59025457855183       
가중치 R2 :  0.6728169219228686
'''