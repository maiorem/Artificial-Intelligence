# 이진분류
import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, MaxPooling1D, Dropout, Conv1D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


dataset=load_breast_cancer()
x=dataset.data
y=dataset.target
# print(x)
# print(y)
# print(x.shape, y.shape) #(569, 30) (569,)

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2)


scaler=MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)



x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],1)


model=Sequential()
model.add(Conv1D(20, kernel_size=2, strides=1, padding='same', input_shape=(30, 1)))
model.add(Conv1D(150, kernel_size=2,  padding='same'))
model.add(Conv1D(100, kernel_size=2, padding='same'))
model.add(Conv1D(80, kernel_size=2, padding='same'))
model.add(Conv1D(70, kernel_size=2, padding='same'))
model.add(Conv1D(50, kernel_size=2, padding='same'))
model.add(Conv1D(20, kernel_size=2, padding='same'))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(20))
model.add(Dense(1, activation='sigmoid'))


model.summary()

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
es=EarlyStopping(monitor='accuracy', patience=50, mode='auto')
# to_hist=TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True)

model.fit(x_train, y_train, epochs=10000, batch_size=1, validation_split=0.2, callbacks=[es])

#4. 평가, 예측
loss, accuracy=model.evaluate(x_test, y_test, batch_size=1)

print('loss : ', loss)
print('accuracy : ', accuracy)


'''
breast cancer LSTM
loss :  0.1184873878955841
accuracy :  0.9473684430122375

breast cancer Conv1D
loss :  0.7208057045936584
accuracy :  0.9736841917037964
'''



