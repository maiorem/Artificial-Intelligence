import numpy as np     

x=np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
            [5,6,7],[6,7,8],[7,8,9],
            [8,9,10],[9,10,11], [10,11,12],
            [20,30,40],[30,40,50],[40,50,60]])
y=np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_input=np.array([50,60,70])

x=x.reshape(13,3,1)
x_input=x_input.reshape(1,3,1)

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, LSTM

model=Sequential()
model.add(LSTM(200, activation='relu', input_shape=(3,1), return_sequences=True))
model.add(LSTM(150, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1))

model.summary()


model.compile(loss='mse', optimizer='adam')


from tensorflow.keras.callbacks import EarlyStopping

early_stopping=EarlyStopping(monitor='loss', patience=50, mode='min')

model.fit(x, y, epochs=10000, batch_size=1, callbacks=[early_stopping])

y_predict=model.predict(x_input)

print("y_predict : ", y_predict)