#체크포인트 불러오기
#모델+가중치

import numpy as np
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
from tensorflow.keras.models import load_model
model=load_model('./model/mnist-08-0.0607.hdf5')

####3. 컴파일, 훈련


####4. 평가, 예측
result=model.evaluate(x_test, y_test, batch_size=32)

print('loss : ', result[0])
print('accuracy : ', result[1])

'''
loss :  0.057394444942474365
accuracy :  0.9825999736785889
'''