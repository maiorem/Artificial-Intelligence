#1. 데이터
import numpy as np   

x1=np.array([range(1, 101), range(311, 411), range(100)])
x2=np.array([range(4, 104), range(761, 861), range(100)])

y1=np.array([range(101, 201), range(711, 811), range(100)])
y2=np.array([range(501, 601), range(431, 531), range(100, 200)])
y3=np.array([range(501, 601), range(431, 531), range(100, 200)])

x1=np.transpose(x1)
x2=np.transpose(x2)
y1=np.transpose(y1)
y2=np.transpose(y2)
y3=np.transpose(y3)

from sklearn.model_selection import train_test_split

x1_train, x1_test, y1_train, y1_test=train_test_split(x1, y1, test_size=0.3)
x2_train, x2_test, y2_train, y2_test, y3_train, y3_test=train_test_split(x2, y2, y3, test_size=0.3)


#2. 모델 구성
from tensorflow.keras.models import Sequential, Model 
from tensorflow.keras.layers import Dense, Input

#모델1
input1=Input(shape=(3,))
dense1=Dense(50, activation='relu', name='dense1')(input1)
dense2=Dense(400, activation='relu', name='dense2')(dense1)
dense3=Dense(30, activation='relu', name='dense3')(dense2)
output1=Dense(3, name='output1')(dense3) 

# model1=Model(inputs=input1, outputs=output1)
# model1.summary()

#모델2
input2=Input(shape=(3,))
dense2_1=Dense(50, activation='relu', name='dense2_1')(input2)
dense2_2=Dense(100, activation='relu', name='dense2_2')(dense2_1)
output2=Dense(3, name='output2')(dense2_2) 

# model2=Model(inputs=input2, outputs=output2)
# model2.summary()


######## 모델 병합(concatenate)
from tensorflow.keras.layers import Concatenate, concatenate
# from keras.layers.merge import Concatenate, concatenate
# from keras.layers import Concatenate, concatenate

# merge1=concatenate([output1, output2])
merge1=Concatenate(axis=1)([output1, output2])
middle1=Dense(30, name='middle1')(merge1)
middle2=Dense(500, name='middle2')(middle1)
middle3=Dense(20, name='middle3')(middle2)

######## output 모델 구성 (분기)
output1_1=Dense(30, name='output1_1')(middle1)
output1_2=Dense(100, name='output1_2')(output1_1)
output1_3=Dense(3, name='output1_3')(output1_2)

output2_1=Dense(15, name='output2_1')(middle1)
output2_2=Dense(200, name='output2_2')(output2_1)
output2_3=Dense(15, name='output2_3')(output2_2)
output2_4=Dense(3, name='output2_4')(output2_3)

output3_1=Dense(100, name='output3_1')(middle1)
output3_2=Dense(500, name='output3_2')(output3_1)
output3_3=Dense(20, name='output3_3')(output3_2)
output3_4=Dense(3, name='output3_4')(output3_3)

######### 모델 정의
model=Model(inputs=[input1, input2], outputs=[output1_3, output2_4, output3_4])

model.summary()


#3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam")
model.fit([x1_train, x2_train], [y1_train, y2_train, y3_train], epochs=100, batch_size=8, 
        validation_split=0.3, verbose=1)



result=model.evaluate([x1_test, x2_test], [y1_test, y2_test, y3_test], batch_size=8)

print("result : ", result)



y1_pred, y2_pred, y3_pred=model.predict([x1_test, x2_test])


from sklearn.metrics import mean_squared_error 
def RMSE(y1_test, y1_pred) :
    return np.sqrt(mean_squared_error(y1_test, y1_pred))

print("RMSE1 : ", RMSE(y1_test, y1_pred))
print("RMSE2 : ", RMSE(y2_test, y2_pred))
print("RMSE3 : ", RMSE(y3_test, y3_pred))

from sklearn.metrics import r2_score 
r2_1=r2_score(y1_test, y1_pred)
r2_2=r2_score(y2_test, y2_pred)
r2_3=r2_score(y3_test, y3_pred)
print("R2_1 : ", r2_1)
print("R2_2 : ", r2_2)
print("R2_3 : ", r2_3)