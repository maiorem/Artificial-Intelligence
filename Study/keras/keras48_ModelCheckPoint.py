#모델 체크포인트 : 훈련을 반복하면서 개선이 생길때마다 저장한다.
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist 

####1.데이터
(x_train, y_train), (x_test, y_test)=mnist.load_data()

x_predict=x_test[:10, :, :]

#인코딩
from tensorflow.keras.utils import to_categorical
y_train=to_categorical(y_train) 
y_test=to_categorical(y_test)


x_train=x_train.reshape(60000,28,28,1).astype('float32')/255. #픽셀의 최대값은 255이므로
x_test=x_test.reshape(10000,28,28,1).astype('float32')/255.

####2. 모델 구성
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

####3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es=EarlyStopping(monitor='val_loss', patience=10, mode='auto')

#모델 체크포인트 저장 경로 설정 : epoch의 두자릿수 정수 - val_loss 소수점 아래 넷째자리
modelpath='./model/{epoch:02d}-{val_loss:.4f}.hdf5'
cp=ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')

hist=model.fit(x_train, y_train, epochs=10000, batch_size=32, verbose=1, validation_split=0.2, callbacks=[es, cp])


# loss=hist.history['loss']
# val_loss=hist.history['val_loss']
# acc = hist.history['accuracy']
# val_acc= hist.history['val_accuracy']


####4. 평가, 예측
result=model.evaluate(x_test, y_test, batch_size=32)

print('loss : ', result[0])
print('accuracy : ', result[1])

##########loss와 acc 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6)) #단위가 무엇인지 찾아볼 것!!!!

plt.subplot(2,1,1) #2행 1열 중 첫번째

plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()

plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')

plt.legend(loc='upper right')


plt.subplot(2,1,2) #2행 1열 중 두번째

plt.plot(hist.history['accuracy'], marker='.', c='red')
plt.plot(hist.history['val_accuracy'], marker='.', c='blue')
plt.grid()

plt.title('accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')

plt.legend(['accuracy', 'val_accuracy'])

plt.show()




x_predict=x_predict.reshape(10, 28, 28,1).astype('float32')/255.


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

