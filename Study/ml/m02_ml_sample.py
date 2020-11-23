######### 머신러닝 예제

import numpy as np
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#머신러닝 임포트
from sklearn.svm import LinearSVC


##### 1. 데이터
dataset=load_iris()
x, y=load_iris(return_X_y=True)
# x=dataset.data
# y=dataset.target
# print(x)
# print(x.shape, y.shape) #(150, 4) (150,)

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2)

# y_train=to_categorical(y_train) 
# y_test=to_categorical(y_test)

scaler=StandardScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)


# #### 2. 모델
# model=Sequential()
# model.add(Dense(80, activation='relu', input_shape=(4,)))
# model.add(Dense(350, activation='relu'))
# model.add(Dropout(0.1))
# model.add(Dense(550, activation='relu'))
# model.add(Dense(1000, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(480, activation='relu'))
# model.add(Dropout(0.4))
# model.add(Dense(80, activation='relu'))
# model.add(Dense(30))
# model.add(Dense(3, activation='softmax'))

model=LinearSVC()
# model.summary()



##### 3. 훈련
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
# es=EarlyStopping(monitor='loss', patience=50, mode='auto')
# to_hist=TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True)

 
# model.fit(x_train, y_train, epochs=10000, batch_size=1, validation_split=0.2)
model.fit(x_train, y_train)

##### 4. 평가, 예측
# loss, accuracy=model.evaluate(x_test, y_test, batch_size=1)

# print('loss : ', loss)
# print('accuracy : ', accuracy)

result=model.score(x_test, y_test)
print("score : ", result) # 0.9666666666666667

# y_predict=model.predict(x_test)
# print('예측값 : ', y_predict)

# y_predict=np.argmax(y_predict, axis=1)
# y_actually=np.argmax(y_test, axis=1)
# print('실제값 : ', y_actually)



'''
1. 원핫 인코딩을 안해도 됨
2. 모델이 간략해짐 (LinearSVC : 선형다중)
'''