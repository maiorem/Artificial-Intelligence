import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
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


# x_train=x_train.reshape(x_train.shape[0], x_train.shape[1],1)
# x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],1)
# x_train=x_train.reshape(x_predict.shape[0], x_predict.shape[1],1)


model=Sequential()
model.add(Dense(80, activation='relu', input_shape=(13,)))
model.add(Dense(250, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(400, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(500, activation='relu'))
model.add(Dense(1800, activation='relu'))
model.add(Dense(780, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(480, activation='relu'))
model.add(Dense(180))
model.add(Dense(80))
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
Standard
RMSE :  4.178725891726597
R2 :  0.7680806961654054
'''

