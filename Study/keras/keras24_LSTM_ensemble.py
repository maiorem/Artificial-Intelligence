# 다음 x1, x2, y 데이터를 LSTM으로 앙상블 모델 만들기. 
# 원하는 결과값은 85, 95. 이게 나오도록 튜닝.

from numpy import array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LSTM, concatenate

x1=array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
            [5,6,7],[6,7,8],[7,8,9],
            [8,9,10],[9,10,11], [10,11,12],
            [20,30,40],[30,40,50],[40,50,60]])
x2=array([[10,20,30],[20,30,40],[30,40,50],[40,50,60],
            [50,60,70],[60,70,80], [70,80,90], [80,90,100],
            [90,100,110], [100,110,120],
            [2,3,4], [3,4,5], [4,5,6]])
y=array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x1_predict=array([55,65,75])
x2_predict=array([65,75,85])

x1_lstm=x1.reshape(13,3,1)
x2_lstm=x2.reshape(13,3,1)

x1_predict=x1_predict.reshape(1,3,1)
x2_predict=x2_predict.reshape(1,3,1)


input1_1=Input(shape=(3,1))
dense1_2=LSTM(100, activation='relu')(input1_1)
dense1_3=Dense(80, activation='relu')(dense1_2)
dense1_4=Dense(30, activation='relu')(dense1_3)
dense1_5=Dense(10, activation='relu')(dense1_4)
output1=Dense(1)(dense1_5)

input2_1=Input(shape=(3,1))
dense2_2=LSTM(150, activation='relu')(input2_1)
dense2_3=Dense(100, activation='relu')(dense2_2)
dense2_4=Dense(50, activation='relu')(dense2_3)
dense2_5=Dense(10, activation='relu')(dense2_4)
output2=Dense(1)(dense2_5)

merge=concatenate([output1, output2])
middle1=Dense(100)(merge)
middle2=Dense(50)(middle1)
output3_1=Dense(20)(middle2)
output3_2=Dense(10)(output3_1)
output3_3=Dense(1)(output3_2)


model=Model(inputs=[input1_1, input2_1], outputs=output3_3)

model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
early_stopping=EarlyStopping(monitor='loss', patience=100, mode='min')

model.fit([x1_lstm, x2_lstm], y, epochs=10000, batch_size=1, callbacks=[early_stopping])

y1_predict=model.predict([x1_predict, x2_predict])
y2_predict=model.predict([x2_predict, x1_predict])

print("y_predict : ", y1_predict, y2_predict)