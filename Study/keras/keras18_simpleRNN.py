# DNN : Deep Neural Network => 은닉 레이어가 2개 이상인 신경망
# RNN : Recurrent Neural Network => 순환신경망. 순차적인 데이터 처리 => Time Series 시계열
# LSTM : RNN에서 가장 성능 좋은 기법

#1. 데이터
import numpy as np 

x=np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]]) #(4,3)
y=np.array([4,5,6,7]) #(4,)

print("x.shape : ", x.shape)
print("y.shape : ", y.shape)

x=x.reshape(x.shape[0], x.shape[1], 1)  #x의 자료를 하나씩 잘라 연산하도록([[[1],[2],[3]], [[2],[3],[4]],[[3],[4],[5]],[[4],[5],[6]]]) / (4,3)=>(4,3,1)로 변환 
                                        #LSTM 행x열x몇개씩 잘라 작업하는지(자르는 크기) 

# from sklearn.model_selection import train_test_split

# x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.3)

#x=x.reshape(4,3,1)
print("x.shape : ", x.shape)


#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN

model=Sequential()
#model.add(LSTM(20, activation='relu', input_shape=(3,1)))   
model.add(SimpleRNN(10, activation='relu', input_length=3, input_dim=1))   
model.add(Dense(30))
model.add(Dense(100))
model.add(Dense(30))
model.add(Dense(1))

model.summary()
#simple RNN params =   = (노드 * 노드) + (input_dim * 노드) + biases  = (10*10)+(1*10) + 10= 120
                    # = (input_dim + 노드) * 노드 + biases = (1 + 10) * 10 + 10 = 120
                


'''
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1, validation_split=0.3)

x_input=np.array([5,6,7]) #(3,) ->(1,3,1)
x_input=x_input.reshape(1,3,1)


# loss=model.evaluate(x_test, y_test, batch_size=1)

y_predict=model.predict(x_input)

# print("loss : ", loss)
print("y_predict : ", y_predict)
'''
