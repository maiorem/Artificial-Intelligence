from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, LSTM, Dropout
from tensorflow.keras.layers import MaxPooling2D, Flatten
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
x_predict=x_test[:10, :, :, :]

x_train=x_train.astype('float32')/255.
x_test=x_test.astype('float32')/255.
x_predict=x_predict.astype('float32')/255.

y_train=to_categorical(y_train) 
y_test=to_categorical(y_test)


################### 1. load_model ########################

#3. 컴파일, 훈련
from tensorflow.keras.models import load_model
model1 = load_model('./save/model_cifar100.h5')
#4. 평가, 예측
result1 = model1.evaluate(x_test, y_test, batch_size=32)



############## 2. load_model ModelCheckPoint #############

from tensorflow.keras.models import load_model

model2 = load_model('./model/cifar100-04-2.5328.hdf5')
#4. 평가, 예측
result2 = model2.evaluate(x_test, y_test, batch_size=32)


################ 3. load_weights ##################

# 2. 모델
model3 = Sequential()
model3.add(Conv2D(64, (3,3), input_shape=(32,32,3)))
model3.add(MaxPooling2D(2, 2))
model3.add(Conv2D(64, (3, 3), activation='relu'))
model3.add(Dropout(0.2))
model3.add(MaxPooling2D(2, 2))
model3.add(Flatten())
model3.add(Dense(512, activation='relu'))
model3.add(Dropout(0.2))
model3.add(Dense(100, activation='softmax'))

# model.summary()

# 3. 컴파일

model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model3.load_weights('./save/weight_cifar100.h5')

#4. 평가, 예측

result3 = model3.evaluate(x_test, y_test, batch_size=32)


print("모델 저장 loss : ", result1[0])
print("모델 저장 accuracy : ", result1[1])

print("가중치 저장 loss : ", result3[0])
print("가중치 저장 accuracy : ", result3[1])

print("체크포인트 loss : ", result2[0])
print("체크포인트 accuracy : ", result2[1])


'''
모델 저장 loss :  3.5971062183380127   
모델 저장 accuracy :  0.3603000044822693
가중치 저장 loss :  3.5971062183380127 
가중치 저장 accuracy :  0.3603000044822693
체크포인트 loss :  2.4969239234924316  
체크포인트 accuracy :  0.37310001254081726
'''