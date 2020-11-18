# 다중분류
import numpy as np
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, MaxPooling2D, Dropout, Conv2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

dataset=load_iris()
x=dataset.data
y=dataset.target
# print(x)
# print(y)
# print(x.shape, y.shape) #(150, 4) (150,)


x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2)

y_train=to_categorical(y_train) 
y_test=to_categorical(y_test)

scaler=StandardScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)


x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],1,1)
x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],1,1)


model=Sequential()
model.add(Conv2D(3, (1,1), padding='same', input_shape=(4,1,1)))
model.add(Conv2D(10, (1,1), padding='same'))
model.add(Conv2D(20, (2,2), padding='same'))
model.add(Conv2D(30, (2,2), padding='same', strides=2))
model.add(Flatten())
model.add(Dense(20, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.summary()
model.save("./save/keras45_cnn.h5")

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
es=EarlyStopping(monitor='loss', patience=50, mode='auto')
# to_hist=TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True)

model.fit(x_train, y_train, epochs=10000, batch_size=1, validation_split=0.2, callbacks=[es])

#4. 평가, 예측
loss, accuracy=model.evaluate(x_test, y_test, batch_size=1)

print('loss : ', loss)
print('accuracy : ', accuracy)



y_predict=model.predict(x_test)
y_predict=np.argmax(y_predict, axis=1)
y_actually=np.argmax(y_test, axis=1)
print('실제값 : ', y_actually)
print('예측값 : ', y_predict)

'''
load_iris CNN

loss :  0.015413327142596245
accuracy :  1.0
실제값 :  [2 2 0 0 0 2 1 2 1 0 0 0 0 0 
1 0 2 1 0 0 1 2 0 1 2 2 0 0 2 1]       
예측값 :  [2 2 0 0 0 2 1 2 1 0 0 0 0 0 
1 0 2 1 0 0 1 2 0 1 2 2 0 0 2 1] 
'''


# scaler=StandardScaler()
# scaler.fit(x_train)
# x_train=scaler.transform(x_train)
# x_test=scaler.transform(x_test)


# x_train=x_train.reshape(x_train.shape[0], x_train.shape[1], 1, 1)
# x_test=x_test.reshape(x_test.shape[0],x_test.shape[1], 1, 1)


# model=Sequential()
# model.add(Conv2D(10, (2,2), padding='same' ,input_shape=(4, 1, 1)))
# model.add(Conv2D(20, (2,2), padding='same'))
# model.add(Conv2D(70, (2,2), padding='same'))
# model.add(Dropout(0.3))
# model.add(Conv2D(50, (2,2), padding='same'))
# model.add(Conv2D(30, (2,2), padding='same'))
# model.add(Flatten())
# model.add(Dense(80, activation='relu'))
# model.add(Dense(30, activation='relu'))
# model.add(Dense(1))


# model.summary()
# model.save("./save/keras45_cnn.h5")

# model.compile(loss='mse', optimizer='adam')


# early_stopping=EarlyStopping(monitor='loss', patience=50, mode='min')

# model.fit(x_train, y_train, epochs=10000, batch_size=1, validation_split=0.2 ,callbacks=[early_stopping])

# y_predict=model.predict(x_test)


# from sklearn.metrics import mean_squared_error 
# def RMSE(y_test, y_pred) :
#     return np.sqrt(mean_squared_error(y_test, y_pred))

# print("RMSE : ", RMSE(y_test, y_predict))

# from sklearn.metrics import r2_score 
# r2=r2_score(y_test, y_predict)
# print("R2 : ", r2)

# '''
# load_iris cnn
# RMSE :  0.13901853177682577
# R2 :  0.9708162131549566
# '''