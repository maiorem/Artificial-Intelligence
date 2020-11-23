import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

# 1. 데이터
x, y=load_iris(return_X_y=True)
x_train, x_test, y_train, y_test=train_test_split(x, y, random_state=66, train_size=0.8, shuffle=True)

scale=StandardScaler()
scale.fit(x_train)
x_train=scale.transform(x_train)
x_test=scale.transform(x_test)

# 2. 모델
# model=LinearSVC()
# model=SVC()
# model=KNeighborsClassifier()
# model=KNeighborsRegressor()
model=RandomForestClassifier()
# model=RandomForestRegressor()

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
score=model.score(x_test, y_test)

# accuracy_score를 넣어서 비교할 것
# 회귀모델일 경우 r2_score와 비교할 것

y_predict=model.predict(x_test)
acc=accuracy_score(y_test, y_predict)
# r2=r2_score(y_test, y_predict)

print('score :', score)
print('acc :', acc)
# print('r2 : ', r2)

print(y_test[:10], '의 예측 결과 ', y_predict[:10])

'''
LinearSVC()
score : 0.9333333333333333
acc : 0.9333333333333333
[1 1 1 0 1 1 0 0 0 2] 의 예측 결과  [1 1 1 0 1 1 0 0 0 2]

SVC()
score : 0.9333333333333333
acc : 0.9333333333333333
[1 1 1 0 1 1 0 0 0 2] 의 예측 결과  [1 1 1 0 1 1 0 0 0 2]

KNeighborsClassifier()
score : 0.9
acc : 0.9
[1 1 1 0 1 1 0 0 0 2] 의 예측 결과  [1 1 2 0 1 1 0 0 0 2]

RandomForestClassifier()
score : 0.9666666666666667
acc : 0.9666666666666667
[1 1 1 0 1 1 0 0 0 2] 의 예측 결과  [1 1 1 0 1 1 0 0 0 2]

KNeighborsRegressor()
score : 0.9073825503355705
r2 :  0.9073825503355705
[1 1 1 0 1 1 0 0 0 2] 의 예측 결과  [1.  1.2 1.6 0.  1.2 1.  0.  0.  0.  1.6]

RandomForestRegressor()
score : 0.9492969798657718
r2 :  0.9492969798657718
[1 1 1 0 1 1 0 0 0 2] 의 예측 결과  [1.   1.1  1.   0.   1.   1.   0.   0.   0.   1.27]
'''