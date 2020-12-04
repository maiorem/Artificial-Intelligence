import numpy as np
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Dense, Flatten, Input, BatchNormalization, Dropout, Activation
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_predict=x_test[:10, :, :, :]

x_train=x_train.astype('float32')/255.
x_test=x_test.astype('float32')/255.
x_predict=x_predict.astype('float32')/255.

y_train=to_categorical(y_train) 
y_test=to_categorical(y_test)

# 2. 모델
vgg19=VGG19(weights='imagenet', include_top=False, input_shape=(32, 32, 3)) # 14,714,688
vgg19.trainable=False


model=Sequential()
model.add(vgg19)
model.add(Flatten())
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Dense(10, activation='softmax'))


# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

es=EarlyStopping(monitor='val_loss', patience=10, mode='auto')
reduce_lr=ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, verbose=1)
modelpath='./model/vgg19-{epoch:02d}-{val_loss:.4f}.hdf5'
cp=ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')

model.fit(x_train, y_train, epochs=1000, batch_size=32, verbose=1, validation_split=0.2, callbacks=[es, cp, reduce_lr])


#4. 평가, 예측
loss, accuracy=model.evaluate(x_test, y_test, batch_size=32)

print('loss : ', loss)
print('accuracy : ', accuracy)

y_predict=model.predict(x_predict)
y_predict=np.argmax(y_predict, axis=1) #One hot encoding의 decoding은 numpy의 argmax를 사용한다.
y_actually=np.argmax(y_test[:10, :], axis=1)
print('실제값 : ', y_actually)
print('예측값 : ', y_predict)

'''
loss :  1.2744640111923218
accuracy :  0.6100000143051147
실제값 :  [3 8 8 0 6 6 1 6 3 1]
예측값 :  [3 8 9 0 6 4 1 6 4 2]
'''