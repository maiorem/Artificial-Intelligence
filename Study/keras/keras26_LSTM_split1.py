import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM 
from tensorflow.keras.callbacks import EarlyStopping
from keras25_split import split_x 


dataset=np.array(range(1,11))
size=5

#모델을 구성하시오. (fit까지만)

datasets=split_x(dataset, size)
x=datasets[:, :size-1]
y=datasets[:, size-1:]

# print(x)
# print(y)
# print(x.shape)

x=x.reshape(6,4,1)


model=Sequential()
model.add(LSTM(120, input_shape=(4,1)))
model.add(Dense(150, activation='relu'))
model.add(Dense(90, activation='relu'))
model.add(Dense(60))
model.add(Dense(10))
model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer='adam')


early_stopping=EarlyStopping(monitor='loss', patience=50, mode='min')
model.fit(x, y, epochs=10000, batch_size=1, callbacks=[early_stopping])

x_predict=np.array([7,8,9,10])
x_predict=x_predict.reshape(1,4,1)

y_predict=model.predict(x_predict)

print("y_predict : ", y_predict)