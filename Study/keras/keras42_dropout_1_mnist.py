#분류

import numpy as np
#OneHotEncoding
#sklearn one hot encoding : 하나의 값만 True, 나머지는 모두 False
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist 

(x_train, y_train), (x_test, y_test)=mnist.load_data()

# print(x_train.shape, x_test.shape) #(60000,28,28), (10000,28,28)
# print(y_train.shape, y_test.shape) #(60000,) (10000,)

x_predict=x_test[:10, :, :]
# print(x_predict.shape) #(10,28,28)
# print(x_predict)



# print(x_train[500])
# print(y_train[500])

# plt.imshow(x_train[500], 'Blues')
# plt.show()

#1. 데이터
#다중 분류 데이터 전처리 (1).OneHotEncoding
from tensorflow.keras.utils import to_categorical
y_train=to_categorical(y_train) #총 10개의 인덱스 카테고리가 있으므로 column은 10이 된다.
y_test=to_categorical(y_test)

# print(y_train.shape, y_test.shape) #(60000,10), (10000,10)
# print(y_train[0])

#MinMaxScaler의 효과를 주는 형변환
x_train=x_train.reshape(60000,28,28,1).astype('float32')/255. #픽셀의 최대값은 255이므로
x_test=x_test.reshape(10000,28,28,1).astype('float32')/255.

# print(x_train[0])

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import Dropout


model=Sequential()
model.add(Conv2D(3, (2,2), padding='same', input_shape=(28,28,1)))
model.add(Dropout(0.2))
model.add(Conv2D(10, (2,2), padding='valid'))
model.add(Dropout(0.2))
model.add(Conv2D(20, (3,3)))
model.add(Dropout(0.2))
model.add(Conv2D(30, (2,2), strides=2))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax')) #softmax=분류값
#(2). 다중 분류의 output layer의 활성화함수는 softmax를 쓴다.

model.summary()

#3. 컴파일, 훈련
#(3). 다중분류에선 반드시 loss를 categorical_crossentropy로 쓴다. 이걸로 이제 accuracy를 잡아줄 수 있다.
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
es=EarlyStopping(monitor='loss', patience=50, mode='auto')
# to_hist=TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True)

model.fit(x_train, y_train, epochs=10000, batch_size=32, verbose=1, validation_split=0.2, callbacks=[es])

#4. 평가, 예측
loss, accuracy=model.evaluate(x_test, y_test, batch_size=32)

print('loss : ', loss)
print('accuracy : ', accuracy)

'''
실습 1. 테스트 데이터를 10개 가져와서 predict 만들 것 (원핫 인코딩 원복할 것)
실습 2. 모델에 es, tensorboard 적용. 
'''

x_predict=x_predict.reshape(10, 28, 28,1).astype('float32')/255.
# print(x_predict)


y_predict=model.predict(x_predict)
y_predict=np.argmax(y_predict, axis=1) #One hot encoding의 decoding은 numpy의 argmax를 사용한다.
y_actually=np.argmax(y_test[:10, :], axis=1)
print('실제값 : ', y_actually)
print('예측값 : ', y_predict)


'''
CNN
loss :  0.20520083606243134
accuracy :  0.983299970626831
실제값 :  [7 2 1 0 4 1 4 9 5 9]        
예측값 :  [7 2 1 0 4 1 4 9 5 9] 
'''

