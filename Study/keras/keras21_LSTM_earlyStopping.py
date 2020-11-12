import numpy as np     

x=np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
            [5,6,7],[6,7,8],[7,8,9],
            [8,9,10],[9,10,11], [10,11,12],
            [20,30,40],[30,40,50],[40,50,60]])
y=np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_input=np.array([50,60,70])

x=x.reshape(13,3,1)
x_input=x_input.reshape(1,3,1)

from tensorflow.keras.models import Sequential, Model 
from tensorflow.keras.layers import Dense, LSTM, Input

# model=Sequential()
# model.add(LSTM(40, activation='relu', input_length=3, input_dim=1))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(500, activation='relu'))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(80, activation='relu'))
# model.add(Dense(20, activation='relu'))
# model.add(Dense(1))

input1=Input(shape=(3,1))
layer1=LSTM(200, activation='relu')(input1)
layer2=Dense(150, activation='relu')(layer1)
layer3=Dense(80, activation='relu')(layer2)
layer4=Dense(50, activation='relu')(layer3)
layer5=Dense(10)(layer4)
output1=Dense(1)(layer5)

model=Model(inputs=input1, outputs=output1)
model.summary()

model.compile(loss='mse', optimizer='adam')

#조기종료
from tensorflow.keras.callbacks import EarlyStopping
#early_stopping=EarlyStopping(monitor='loss', patience=100, mode='min')
early_stopping=EarlyStopping(monitor='loss', patience=50, mode='auto')

model.fit(x, y, epochs=10000, batch_size=1, callbacks=[early_stopping])

y_predict=model.predict(x_input)

print("y_predict : ", y_predict)