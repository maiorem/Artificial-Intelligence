#분류

import numpy as np
#OneHotEncoding
#sklearn one hot encoding : 하나의 값만 True, 나머지는 모두 False
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist 

(x_train, y_train), (x_test, y_test)=mnist.load_data()

print(x_train.shape, x_test.shape) #(60000,24,28), (10000,28,28)
print(y_train.shape, y_test.shape) #(60000,) (10000,)
# print(x_train[500])
# print(y_train[500])

# plt.imshow(x_train[500], 'Blues')
# plt.show()

#1. 데이터
#다중 분류 데이터 전처리 (1).OneHotEncoding
from tensorflow.keras.utils import to_categorical
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

print(y_train.shape, y_test.shape) #(60000,10), (10000,10)
print(y_train[0])

#MinMaxScaler의 효과를 주는 형변환
x_train=x_train.reshape(60000,28,28,1).astype('float32')/255.
x_test=x_test.reshape(10000,28,28,1).astype('float32')/255.

print(x_train[0])

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

model=Sequential()
model.add(Conv2D(10, (2,2), padding='same', input_shape=(28,28,1)))
model.add(Conv2D(20, (2,2), padding='valid'))
model.add(Conv2D(30, (3,3)))
model.add(Conv2D(40, (2,2), strides=2))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='softmax')) #softmax=분류값
#(2). 다중 분류의 output layer의 활성화함수는 softmax를 쓴다.

model.summary()

#3. 컴파일, 훈련
#(3). 다중분류에선 반드시 loss를 categorical_crossentropy로 쓴다. 이걸로 이제 accuracy를 잡아줄 수 있다.
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=30, batch_size=32, verbose=1, validation_split=0.2)

#4. 평가, 예측
loss, accuracy=model.evaluate(x_test, y_test, batch_size=32)

print('loss : ', loss)
print('accuracy : ', accuracy)