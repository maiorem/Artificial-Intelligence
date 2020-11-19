from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv1D, LSTM
from tensorflow.keras.layers import MaxPooling1D, Flatten
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
x_predict=x_test[:10, :, :, :]


x_train=x_train.reshape(50000, 32*32, 3).astype('float32')/255.
x_test=x_test.reshape(10000, 32*32, 3).astype('float32')/255.
x_predict=x_predict.reshape(10, 32*32, 3).astype('float32')/255.


y_train=to_categorical(y_train) 
y_test=to_categorical(y_test)


model=Sequential()
model.add(Conv1D(10, kernel_size=2, strides=1, padding='same', input_shape=(32*32, 3)))
model.add(Conv1D(30, kernel_size=2,  padding='same'))
model.add(Conv1D(20, kernel_size=2, padding='same'))
model.add(Flatten())
model.add(Dense(50))
model.add(Dense(100, activation='softmax'))  

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
es=EarlyStopping(monitor='loss', patience=3, mode='auto')
# to_hist=TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True)

model.fit(x_train, y_train, epochs=50, batch_size=2000, verbose=1, validation_split=0.2, callbacks=[es])

#4. 평가, 예측
loss, accuracy=model.evaluate(x_test, y_test, batch_size=2000)

print('loss : ', loss)
print('accuracy : ', accuracy)


y_predict=model.predict(x_predict)
y_predict=np.argmax(y_predict, axis=1)
y_actually=np.argmax(y_test[:10, :], axis=1)
print('실제값 : ', y_actually)
print('예측값 : ', y_predict)

'''
cifar100 LSTM
---메모리 폭파---- 

cifar100 Conv1D
loss :  3.885857105255127
accuracy :  0.14499999582767487
실제값 :  [49 33 72 51 71 92 15 14 23  0]
예측값 :  [71 75 93 10 70 74 28 74 71 10]
'''