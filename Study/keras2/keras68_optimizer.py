import numpy as np

#1. 데이터
x=np.array([1,2,3,4,5,6,7,8,9,10])
y=np.array([1,2,3,4,5,6,7,8,9,10])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#2. 모델 구성
model=Sequential()
model.add(Dense(300, input_dim=1)) 
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad, RMSprop, SGD, Nadam

# optimizer = Adam(lr=0.1)  # lr=0.001 => loss :  [4.581579473444952e-12, 4.581579473444952e-12] 예측 결과물 :  [[10.999996]]
                            # lr=0.0001 => loss :  [0.07129351794719696, 0.07129351794719696] 예측 결과물 :  [[10.647432]]
                            # lr=0.01 => loss :  [1.7735717392497463e-06, 1.7735717392497463e-06] 예측 결과물 :  [[10.997658]]
                            # lr=0.1 => loss :  [2.8946498787263408e-05, 2.8946498787263408e-05] 예측 결과물 :  [[11.009992]]
# optimizer = Adadelta(lr=0.18) # lr=0.001 => loss :  [38.526512145996094, 38.526512145996094] 예측 결과물 :  [[-0.01599766]]
                                # lr=0.1 => loss :  [0.0013681253185495734, 0.0013681253185495734] 예측 결과물 :  [[10.93656]]
                                # lr=0.15 => loss :  [4.058718332089484e-06, 4.058718332089484e-06] 예측 결과물 :  [[10.995738]]
                                # lr=0.18 => loss :  [8.425615760643268e-11, 8.425615760643268e-11] 예측 결과물 :  [[10.999986]]
# optimizer = Adamax(lr=0.005)  # lr=0.001 => loss :  [0.007927613332867622, 0.007927613332867622] 예측 결과물 :  [[10.896457]]
                                # lr=0.005 => loss :  [8.462350820082065e-10, 8.462350820082065e-10] 예측 결과물 :  [[10.999957]]
optimizer = Adagrad(lr=0.01)    # lr=0.001 => loss :  [0.048510242253541946, 0.048510242253541946] 예측 결과물 :  [[10.712738]]
                                # lr=0.01 => loss :  [1.476237743158748e-10, 1.476237743158748e-10] 예측 결과물 :  [[10.9999895]]
# optimizer = RMSprop(lr=0.001) # loss :  [2.4399958419962786e-05, 2.4399958419962786e-05] 예측 결과물 :  [[10.994353]]
# optimizer = SGD(lr=0.001)   # loss :  [1.6223628335865214e-05, 1.6223628335865214e-05] 예측 결과물 :  [[10.992418]]
# optimizer = Nadam(lr=0.001)  # loss :  [3.1621268135495484e-05, 3.1621268135495484e-05] 예측 결과물 :  [[10.992557]]

model.compile(loss='mse', optimizer=optimizer, metrics=['mse']) 
model.fit(x, y, epochs=100, batch_size=1) 


#4. 평가, 예측
loss=model.evaluate(x, y, batch_size=1)
y_pred=model.predict([11])

print("loss : ", loss, '예측 결과물 : ', y_pred)


