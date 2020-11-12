# 만약에 여러개의 레이어에 LSTM을 쌓는다면?
# LSTM은 2차원을 3차원으로 형변환하여 입력하였으므로 3차원이지만 return_sequences가 디폴트로 False로 주어져 있기에 마지막 출력 시퀀스에 맞춘 차원을 반환한다.
# 때문에 LSTM이 한번 더 나오면 요구되는 입력값이 3차원인데 2차원만 입력되는 경우가 발생함.
# 그럴 땐 앞에 쓴 LSTM 함수의 return_sequence를 True로 바꿔 본연의 차원인 3차원을 되찾고, 그대로 입력하도록 만든다. 
# 하지만 LSTM을 중복해서 쓰면 성능이 떨어짐.....  이런 게 있다는 건 정리해두자.

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