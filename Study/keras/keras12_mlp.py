#1. 데이터
import numpy as np   

#(100,3) 데이터 만들기
x=np.array([range(1, 101), range(311, 411), range(100)])
x=np.transpose(x)
y=np.array([range(101, 201), range(711, 811), range(100)])
y=np.transpose(y)

print(x)
print(x.shape) 

#모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model=Sequential()

