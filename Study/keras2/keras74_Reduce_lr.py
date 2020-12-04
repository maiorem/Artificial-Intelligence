import numpy as np
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist 

(x_train, y_train), (x_test, y_test)=mnist.load_data()


x_predict=x_test[:10, :, :]

#1. 데이터
from tensorflow.keras.utils import to_categorical
y_train=to_categorical(y_train) 
y_test=to_categorical(y_test)


x_train=x_train.reshape(60000,28,28,1).astype('float32')/255. #픽셀의 최대값은 255이므로
x_test=x_test.reshape(10000,28,28,1).astype('float32')/255.

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

model=Sequential()
model.add(Conv2D(3, (2,2), padding='same', input_shape=(28,28,1)))
model.add(Conv2D(10, (2,2), padding='valid'))
model.add(Conv2D(20, (3,3)))
model.add(Conv2D(30, (2,2), strides=2))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='softmax')) 

model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

es=EarlyStopping(monitor='val_loss', patience=6, mode='auto')
ck=ModelCheckpoint('./model/keras74-{epoch:02d}-{val_loss:.4f}.hdf5', save_weights_only=True, save_best_only=True, monitor='val_loss', mode='min', verbose=1)

# learning_rate를 감소시켜 주는 콜백 함수 // 3번까지도(patience) 개선이 없으면 50퍼센트 감소(factor). 얼리스타핑이 다 지날때까지도 개선이 없으면 종료.
reduce_lr=ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, verbose=1)

model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1, validation_split=0.2, callbacks=[es, ck, reduce_lr])

#4. 평가, 예측
loss, accuracy=model.evaluate(x_test, y_test, batch_size=32)

print('loss : ', loss)
print('accuracy : ', accuracy)

x_predict=x_predict.reshape(10, 28, 28,1).astype('float32')/255.
# print(x_predict)


y_predict=model.predict(x_predict)
y_predict=np.argmax(y_predict, axis=1) 
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

