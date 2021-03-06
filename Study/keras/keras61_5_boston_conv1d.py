import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
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


x_train=x_train.reshape(x_train.shape[0], x_train.shape[1],1)
x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],1)

model=Sequential()
model.add(Conv1D(20, kernel_size=2, strides=1, padding='same', input_shape=(13, 1)))
model.add(Conv1D(150, kernel_size=2,  padding='same'))
model.add(Conv1D(100, kernel_size=2, padding='same'))
model.add(Conv1D(80, kernel_size=2, padding='same'))
model.add(Conv1D(70, kernel_size=2, padding='same'))
model.add(Conv1D(50, kernel_size=2, padding='same'))
model.add(Conv1D(20, kernel_size=2, padding='same'))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(20))
model.add(Dense(1))


model.summary()

model.compile(loss='mse', optimizer='adam')


early_stopping=EarlyStopping(monitor='loss', patience=50, mode='min')

model.fit(x_train, y_train, epochs=10000, batch_size=1, validation_split=0.2 ,callbacks=[early_stopping])

y_predict=model.predict(x_test)


from sklearn.metrics import mean_squared_error 
def RMSE(y_test, y_pred) :
    return np.sqrt(mean_squared_error(y_test, y_pred))

print("RMSE : ", RMSE(y_test, y_predict))

from sklearn.metrics import r2_score 
r2=r2_score(y_test, y_predict)
print("R2 : ", r2)


'''
boston LSTM
RMSE :  3.4722069969249505
R2 :  0.8554447018389687

boston Conv1D
RMSE :  4.3295958833183485
R2 :  0.8015621993977303
'''