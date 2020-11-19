# 다중분류
import numpy as np
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, MaxPooling2D, Dropout, Conv2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#데이터 불러오기
x=np.load('./data/iris_x.npy')
y=np.load('./data/iris_y.npy')

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2)


y_train=to_categorical(y_train) 
y_test=to_categorical(y_test)

scaler=StandardScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)


x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],1,1)
x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],1,1)


################### 1. load_model ########################

#3. 컴파일, 훈련
from tensorflow.keras.models import load_model
model1 = load_model('./save/model_iris.h5')
#4. 평가, 예측
result1 = model1.evaluate(x_test, y_test, batch_size=32)



############## 2. load_model ModelCheckPoint #############

from tensorflow.keras.models import load_model

model2 = load_model('./model/iris-134-0.0097.hdf5')
#4. 평가, 예측
result2 = model2.evaluate(x_test, y_test, batch_size=32)


################ 3. load_weights ##################

# 2. 모델
model3 = Sequential()
model3.add(Conv2D(3, (1,1), padding='same', input_shape=(4,1,1)))
model3.add(Conv2D(10, (1,1), padding='same'))
model3.add(Conv2D(20, (2,2), padding='same'))
model3.add(Conv2D(30, (2,2), padding='same', strides=2))
model3.add(Flatten())
model3.add(Dense(20, activation='relu'))
model3.add(Dense(3, activation='softmax'))

# model.summary()

# 3. 컴파일

model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model3.load_weights('./save/weight_iris.h5')

#4. 평가, 예측

result3 = model3.evaluate(x_test, y_test, batch_size=32)


print("모델 저장 loss : ", result1[0])
print("모델 저장 accuracy : ", result1[1])

print("가중치 저장 loss : ", result3[0])
print("가중치 저장 accuracy : ", result3[1])

print("체크포인트 loss : ", result2[0])
print("체크포인트 accuracy : ", result2[1])


'''
모델 저장 loss :  0.018727337941527367 
모델 저장 accuracy :  1.0
가중치 저장 loss :  0.018727337941527367
가중치 저장 accuracy :  1.0
체크포인트 loss :  0.008886545896530151
체크포인트 accuracy :  1.0
'''