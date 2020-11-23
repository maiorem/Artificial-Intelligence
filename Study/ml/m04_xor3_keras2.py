# XOR 게이트  : Deep Learning => 연산을 늘리면 해결됨

from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x_data=[[0,0], [0,1], [1,0], [1,1]]
y_data=[0,1,1,0]

#2. 모델
# model=LinearSVC()
# model=SVC()
model=Sequential()
model.add(Dense(20, activation='relu', input_dim=2))
model.add(Dense(100, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_data, y_data, epochs=100, batch_size=1)

# 4. 평가, 예측
y_predict=model.predict(x_data)
print(x_data, '의 예측 결과', y_predict)

acc1=model.evaluate(x_data, y_data, batch_size=1)
print('model.score : ',acc1)

# acc2=accuracy_score(y_data, y_predict)
# print('accuracy_score : ',acc2)