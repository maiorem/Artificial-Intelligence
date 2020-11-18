# 이진분류
import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, MaxPooling2D, Dropout, Conv2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
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

model=Sequential()
model.add(Dense(80, activation='relu', input_shape=(30,)))
model.add(Dense(350, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(550, activation='relu'))
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(480, activation='relu'))
model.add(Dense(180, activation='relu'))
model.add(Dense(80))
model.add(Dense(1, activation='sigmoid'))


model.summary()
model.save("./save/keras46_dnn.h5")
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
es=EarlyStopping(monitor='accuracy', patience=30, mode='auto')
# to_hist=TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True)
#모델 체크포인트 저장 경로 설정 : epoch의 두자릿수 정수 - val_loss 소수점 아래 넷째자리
modelpath='./model/cancer-{epoch:02d}-{val_loss:.4f}.hdf5'
cp=ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
hist=model.fit(x_train, y_train, epochs=10000, batch_size=1, validation_split=0.2, callbacks=[es, cp])

#4. 평가, 예측
loss, accuracy=model.evaluate(x_test, y_test, batch_size=1)

print('loss : ', loss)
print('accuracy : ', accuracy)



# y_predict=model.predict(x_test)

# print('실제값 : ', y_test)
# print('예측값 : ', y_predict)

'''
breast cancer DNN

loss :  0.043071404099464417
accuracy :  0.9912280440330505
'''

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
