import numpy as np
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

####1.데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_predict=x_test[:10, :, :, :]

x_train=x_train.astype('float32')/255.
x_test=x_test.astype('float32')/255.
x_predict=x_predict.astype('float32')/255.

y_train=to_categorical(y_train) 
y_test=to_categorical(y_test)


################### 1. load_model ########################

#3. 컴파일, 훈련
from tensorflow.keras.models import load_model
model1 = load_model('./save/model_cifar10.h5')
#4. 평가, 예측
result1 = model1.evaluate(x_test, y_test, batch_size=32)



############## 2. load_model ModelCheckPoint #############

from tensorflow.keras.models import load_model

model2 = load_model('./model/cifar10-04-1.1035.hdf5')
#4. 평가, 예측
result2 = model2.evaluate(x_test, y_test, batch_size=32)


################ 3. load_weights ##################

# 2. 모델
model3 = Sequential()
model3.add(Conv2D(3, (2,2), input_shape=(32,32,3)))
model3.add(Conv2D(20, (2,2)))
model3.add(Conv2D(30, (2,2)))
model3.add(Conv2D(50, (2,2)))
model3.add(Conv2D(70, (2,2)))
model3.add(MaxPooling2D(pool_size=2))
model3.add(Flatten())
model3.add(Dense(20, activation='relu'))
model3.add(Dense(10, activation='softmax')) 

# model.summary()

# 3. 컴파일

model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model3.load_weights('./save/weight_cifar10.h5')

#4. 평가, 예측

result3 = model3.evaluate(x_test, y_test, batch_size=32)


print("모델 저장 loss : ", result1[0])
print("모델 저장 accuracy : ", result1[1])

print("가중치 저장 loss : ", result3[0])
print("가중치 저장 accuracy : ", result3[1])

print("체크포인트 loss : ", result2[0])
print("체크포인트 accuracy : ", result2[1])


'''
모델 저장 loss :  1.891913890838623    
모델 저장 accuracy :  0.5956000089645386
가중치 저장 loss :  1.891913890838623  
가중치 저장 accuracy :  0.5956000089645386
체크포인트 loss :  1.1154990196228027  
체크포인트 accuracy :  0.6208999752998352
'''