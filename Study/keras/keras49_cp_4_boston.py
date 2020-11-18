import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

dataset=load_boston()
x=dataset.data
y=dataset.target
# print(x)
# print(x.shape, y.shape) #(506, 13) (506,)



x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2)


scaler=MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)


# x_train=x_train.reshape(x_train.shape[0], x_train.shape[1],1)
# x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],1)
# x_train=x_train.reshape(x_predict.shape[0], x_predict.shape[1],1)


model=Sequential()
model.add(Dense(80, activation='relu', input_shape=(13,)))
model.add(Dense(150, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(350, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(700, activation='relu'))
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(480, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(280, activation='relu'))
model.add(Dense(80))
model.add(Dense(30))
model.add(Dense(1))


model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mae'])

#모델 체크포인트 저장 경로 설정 : epoch의 두자릿수 정수 - val_loss 소수점 아래 넷째자리
modelpath='./model/boston-{epoch:02d}-{val_loss:.4f}.hdf5'
cp=ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
early_stopping=EarlyStopping(monitor='val_loss', patience=10, mode='min')

hist=model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2 ,callbacks=[early_stopping, cp])

y_predict=model.predict(x_test)


from sklearn.metrics import mean_squared_error 
def RMSE(y_test, y_pred) :
    return np.sqrt(mean_squared_error(y_test, y_pred))

print("RMSE : ", RMSE(y_test, y_predict))

from sklearn.metrics import r2_score 
r2=r2_score(y_test, y_predict)
print("R2 : ", r2)

'''
RMSE :  3.054935090533644
R2 :  0.8877071672091482
'''

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6)) #단위가 무엇인지 찾아볼 것!!!!

plt.subplot(2,1,1) #2행 1열 중 첫번째

plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()

plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')

plt.legend(loc='upper right')


plt.subplot(2,1,2) #2행 1열 중 두번째

plt.plot(hist.history['mae'], marker='.', c='red')
plt.plot(hist.history['val_mae'], marker='.', c='blue')
plt.grid()

plt.title('mae')
plt.ylabel('mae')
plt.xlabel('epoch')

plt.legend(['mae', 'val_mae'])

plt.show()