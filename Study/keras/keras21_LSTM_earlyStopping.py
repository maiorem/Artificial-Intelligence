#Early Stopping : 훈련을 시키는 도중, 가장 낮은 loss 값이 나왔을 때 훈련을 멈추고 결과값을 출력하도록 하는 함수. patience 값을 조절하여 낮은 loss보다 더 낮은 loss를 찾기 위해 현재의 저장값을 넘길 수 있다. 

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
#early_stopping=EarlyStopping(monitor='loss', patience=100, mode='min') #monitor를 loss로 설정했기 때문에 mode는 가장 낮은 값을 찾도록 해 줌. 만약 accuracy라면 max일 것. 헷갈리면 auto로 해도 됨.
early_stopping=EarlyStopping(monitor='loss', patience=50, mode='auto')

model.fit(x, y, epochs=10000, batch_size=1, callbacks=[early_stopping]) #여기서 파라미터로 조기종료 설정을 불러온다.

y_predict=model.predict(x_input)

print("y_predict : ", y_predict)