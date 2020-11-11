#1. 데이터
import numpy as np   

x=np.array([range(1, 101), range(311, 411), range(100)])
x1=np.transpose(x)
y=np.array([range(101, 201), range(711, 811), range(100)])
y1=np.transpose(y)

print(x1.shape) #(100,3)

x2=np.array([range(4, 104), range(761, 861), range(100)])
x2=np.transpose(x2)
y2=np.array([range(501, 601), range(431, 531), range(100, 200)])
y2=np.transpose(y2)

from sklearn.model_selection import train_test_split

x1_train, x1_test, y1_train, y1_test=train_test_split(x1, y1, test_size=0.2)
x1_train, x1_val, y1_train, y1_val=train_test_split(x1_train, y1_train, test_size=0.2)

x2_train, x2_test, y2_train, y2_test=train_test_split(x2, y2, test_size=0.2)
x2_train, x2_val, y2_train, y2_val=train_test_split(x2_train, y2_train, test_size=0.2)


#2. 모델 구성
from tensorflow.keras.models import Sequential, Model 
from tensorflow.keras.layers import Dense, Input

input1=Input(shape=(3,))
dense1=Dense(50, activation='relu')(input1)
dense2=Dense(400, activation='relu')(dense1)
dense3=Dense(30, activation='relu')(dense2)
output1=Dense(3)(dense3) #선형회귀이기 때문에 마지막은 linear여야 함
model1=Model(inputs=input1, outputs=output1)

model1.summary()

input2=Input(shape=(3,))
dense2_1=Dense(50, activation='relu')(input2)
dense2_2=Dense(400, activation='relu')(dense2_1)
dense2_3=Dense(30, activation='relu')(dense2_2)
output2=Dense(3)(dense2_3) #선형회귀이기 때문에 마지막은 linear여야 함
model2=Model(inputs=input2, outputs=output2)

model2.summary()

