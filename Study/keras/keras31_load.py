import numpy as np
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, LSTM, Input
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from keras25_split import split_x 


dataset=np.array(range(1,101))
size=5

datasets=split_x(dataset, size)
x=datasets[:, :size-1]
y=datasets[:, size-1:]
x=x.reshape(x.shape[0], x.shape[1], 1)

#train과 test 데이터로 가르기
x_train, x_test, y_train, y_test=train_test_split(x, y, train_size=0.7)


# #2. 모델
# model=Sequential()

# model.add(LSTM(100, input_shape=(4,1)))
# model.add(Dense(50))
# model.add(Dense(10))
# model.add(Dense(1))
model=load_model('./save/keras30.h5')
input1 = Input(shape=(4, 1))
dense = model(input1)
output1 = Dense(1)(dense)
model = Model(inputs=input1, outputs=output1)

model.summary()



model.compile(loss='mse', optimizer='adam')


early_stopping=EarlyStopping(monitor='loss', patience=40, mode='min')
model.fit(x_train, y_train, epochs=10000, batch_size=1, verbose=2, callbacks=[early_stopping])
loss=model.evaluate(x_test, y_test, batch_size=1)

x_predict=np.array([97,98,99,100])
x_predict=x_predict.reshape(1,4,1)

y_predict=model.predict(x_predict)


print("y_predict : ", y_predict)
print("loss : ", loss)