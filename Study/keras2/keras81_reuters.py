from tensorflow.keras.datasets import reuters
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

(x_train, y_train), (x_test, y_test)=reuters.load_data(
    num_words=10000, test_split=0.2
)

print(x_train.shape, x_test.shape) #(8982,) (2246,)
print(y_train.shape, y_test.shape) #(8982,) (2246,)

print(x_train[0])
print(y_train[0])

x_train=pad_sequences(x_train, maxlen=1000, padding='pre') # 뒤 : post
x_test=pad_sequences(x_test, maxlen=1000,padding='pre')

print(x_train.shape, x_test.shape) # (8982, 1000) (2246, 1000)

print(len(x_train[0])) # 1000
print(len(x_train[11])) # 1000

# y의 카테고리 갯수 출력
category = np.max(y_train)+1
print('카테고리 : ', category)
# 카테고리 :  46

# y의 유니크한 값들 출력
y_bunpo = np.unique(y_train)
print(y_bunpo)
'''
[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45]
'''

y_train=to_categorical(y_train, num_classes=46)
y_test=to_categorical(y_test, num_classes=46)

print(y_train.shape, y_test.shape) #(8982, 46) (2246, 46)



#2. 모델
model=Sequential()
model.add(Embedding(10000, 100, input_length=1000))
model.add(LSTM(32))
model.add(Dense(100))
model.add(Dense(46, activation='softmax'))


model.summary()


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es=EarlyStopping(monitor='val_loss', patience=10, mode='auto')
model.fit(x_train, y_train, epochs=100, validation_split=0.2, callbacks=[es])

loss, acc =model.evaluate(x_test, y_test)

print('loss :', loss)
print('acc : ', acc)

'''
loss : 2.8254828453063965
acc :  0.6687444448471069
'''

