import numpy as np
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout, BatchNormalization, Activation
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_predict=x_test[:10, :, :, :]

x_train=x_train.astype('float32')/255.
x_test=x_test.astype('float32')/255.
x_predict=x_predict.astype('float32')/255.

y_train=to_categorical(y_train) 
y_test=to_categorical(y_test)


model=Sequential()
model.add(Conv2D(64, (3,3), input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(128, (3,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dense(256))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.summary()

#모델 저장
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
es=EarlyStopping(monitor='val_loss', patience=2, mode='auto')
reduce_lr=ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)

model.fit(x_train, y_train, epochs=1000, batch_size=32, verbose=1, validation_split=0.2, callbacks=[es, reduce_lr])


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
loss :  1.1739773750305176
accuracy :  0.6114000082015991
실제값 :  [3 8 8 0 6 6 1 6 3 1]
예측값 :  [8 8 8 0 6 6 1 6 3 1]
'''