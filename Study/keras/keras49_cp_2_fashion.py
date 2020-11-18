from tensorflow.keras.datasets import cifar10, fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, LSTM
from tensorflow.keras.layers import MaxPooling2D, Flatten
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_predict=x_test[:10, :, :]


y_train=to_categorical(y_train) 
y_test=to_categorical(y_test)

x_train=x_train.reshape(60000,28,28,1).astype('float32')/255.
x_test=x_test.reshape(10000,28,28,1).astype('float32')/255.
x_predict=x_predict.reshape(10, 28, 28,1).astype('float32')/255.


model=Sequential()
model.add(Conv2D(3, (2,2), input_shape=(28,28,1)))
model.add(Conv2D(10, (2,2)))
model.add(Conv2D(20, (3,3)))
model.add(Conv2D(30, (2,2), strides=2))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es=EarlyStopping(monitor='val_loss', patience=10, mode='auto')
# to_hist=TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True)
#모델 체크포인트 저장 경로 설정 : epoch의 두자릿수 정수 - val_loss 소수점 아래 넷째자리
modelpath='./model/fashion-{epoch:02d}-{val_loss:.4f}.hdf5'
cp=ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')

hist=model.fit(x_train, y_train, epochs=1000, batch_size=32, verbose=1, validation_split=0.2, callbacks=[es, cp])

#4. 평가, 예측
loss, accuracy=model.evaluate(x_test, y_test, batch_size=32)

print('loss : ', loss)
print('accuracy : ', accuracy)



y_predict=model.predict(x_predict)
y_predict=np.argmax(y_predict, axis=1)
y_actually=np.argmax(y_test[:10, :], axis=1)
print('실제값 : ', y_actually)
print('예측값 : ', y_predict)

'''
fashion_mnist CNN
loss :  0.5424944162368774
accuracy :  0.9741999864578247
실제값 :  [9 2 1 1 6 1 4 6 5 7]        
예측값 :  [9 2 1 1 6 1 4 6 5 7]
'''

##########loss와 acc 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6)) #(가로, 세로) inch

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