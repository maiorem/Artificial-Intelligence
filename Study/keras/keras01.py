import numpy as np

#1. 데이터
# => 데이터는 건들지 않음
x=np.array([1,2,3,4,5])
y=np.array([1,2,3,4,5])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#2. 모델 구성
# => 노드와 레이어로 튜닝
model=Sequential()
model.add(Dense(100000, input_dim=1)) #1차원 배열이므로 1차원 입력(input_dim=1)
model.add(Dense(2))
model.add(Dense(1)) #1차원이므로 출력도 1


#3. 컴파일, 훈련
# => 데이터인 x, y는 그대로 두고 나머지 튜닝 : 하이퍼 파라미터 튜닝
model.compile(loss='mse', optimizer='adam', metrics=['acc']) #컴파일 : 손실(loss)은 mse로 처리, 최적화(optimaizer)는 'adam'으로, 매트릭스는 accuracy
#model.fit(x, y, epochs=1000, batch_size=1) #데이터 하나씩(batch_size=1 / 디폴트는 32) 1000번(epochs) 훈련시켜라
model.fit(x, y, epochs=1000)

#4. 평가, 예측
#loss, acc=model.evaluate(x, y, batch_size=1)
loss, acc=model.evaluate(x, y)
## 훈련시킨 데이터로 평가를 하는 오류

print("loss : ", loss)
print("acc : ", acc)

#acc 값이 1.0이 나오도록 튜닝할 것

