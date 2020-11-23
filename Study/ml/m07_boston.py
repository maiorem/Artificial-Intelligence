import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

# 1. 데이터
x, y=load_boston(return_X_y=True)
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
# model=RandomForestClassifier()
model=RandomForestRegressor()

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
score=model.score(x_test, y_test)

# accuracy_score를 넣어서 비교할 것
# 회귀모델일 경우 r2_score와 비교할 것

y_predict=model.predict(x_test)
# acc=accuracy_score(y_test, y_predict)
r2=r2_score(y_test, y_predict)

print('score :', score)
# print('acc :', acc)
print('r2 : ', r2)

print(y_test[:10], '의 예측 결과 ', y_predict[:10])

'''
LinearSVC() 
: ValueError: Unknown label type: 'continuous'

SVC() 
: ValueError: Unknown label type: 'continuous'

KNeighborsClassifier()
: ValueError: Unknown label type: 'continuous'

RandomForestClassifier()
: ValueError: Unknown label type: 'continuous'

KNeighborsRegressor()
score : 0.8404010032786686
r2 :  0.8404010032786686
[16.3 43.8 24.  50.  20.5 19.9 17.4 21.8 41.7 13.1] 의 예측 결과  [12.56 38.66 23.44 42.88 19.8  20.36 22.8  20.68 42.48 18.44] 

RandomForestRegressor() 
score : 0.9253825125844518
r2 :  0.9253825125844518
[16.3 43.8 24.  50.  20.5 19.9 17.4 21.8 41.7 13.1] 의 예측 결과  [15.346 45.621 28.078 47.088 21.098 21.114 19.902 20.528 45.064 16.41 ]
'''