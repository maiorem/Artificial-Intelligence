import numpy as np
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist 

(x_train, y_train), (x_test, y_test)=mnist.load_data()

# print(x_train.shape, x_test.shape) #(60000,28,28), (10000,28,28)
# print(y_train.shape, y_test.shape) #(60000,) (10000,)

x_predict=x_test[:10, :, :]

# print(x_predict.shape) #(10,28,28)
# print(x_predict)


#1. 데이터
from tensorflow.keras.utils import to_categorical
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)


x_train=x_train.reshape(60000,28*7, 4).astype('float32')/255. #(60000, 28x28)
x_test=x_test.reshape(10000,28*7, 4).astype('float32')/255.



#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LSTM, Dropout

model=Sequential()
model.add(Conv1D(64, kernel_size=2, strides=1, padding='same', input_shape=(28*7, 4)))
model.add(Conv1D(32, kernel_size=2,  padding='same'))
model.add(Conv1D(32, kernel_size=2, padding='same'))
model.add(Dropout(0.2))
model.add(Conv1D(16, kernel_size=2, padding='same'))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(20))
model.add(Dense(1))
model.add(Dense(10, activation='softmax')) 

model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
es=EarlyStopping(monitor='loss', patience=10, mode='auto')

model.fit(x_train, y_train, epochs=1000, batch_size=128, verbose=1, validation_split=0.2, callbacks=[es])

#4. 평가, 예측
loss, accuracy=model.evaluate(x_test, y_test, batch_size=128)

print('loss : ', loss)
print('accuracy : ', accuracy)



x_predict=x_predict.reshape(10, 28*7, 4).astype('float32')/255.


y_predict=model.predict(x_predict)
y_predict=np.argmax(y_predict, axis=1)
y_actually=np.argmax(y_test[:10, :], axis=1)
print('실제값 : ', y_actually)
print('예측값 : ', y_predict)


'''
mnist LSTM
loss :  0.3592512804985046
accuracy :  0.8765999979972839
실제값 :  [7 2 1 0 4 1 4 9 5 9]        
예측값 :  [7 2 1 0 4 1 4 9 5 9]

mnist Conv1D
loss :  1.2144277095794678
accuracy :  0.5551999807357788
실제값 :  [7 2 1 0 4 1 4 9 5 9]
예측값 :  [7 0 1 0 9 1 1 9 0 7]
'''

