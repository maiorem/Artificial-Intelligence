import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

# 1. 데이터
x, y=load_diabetes(return_X_y=True)
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
r2=r2_score(y_test, y_predict)

print('score :', score)
print('acc :', acc)
print('r2 : ', r2)

print(y_test[:10], '의 예측 결과 ', y_predict[:10])

'''
LinearSVC()
score : 0.011235955056179775
acc : 0.011235955056179775
[235. 150. 124.  78. 168. 253.  97. 102. 249. 142.] 의 예측 결과  [ 91. 232. 281. 134.  97.  69.  53. 109. 230. 101.]

SVC()
score : 0.0
acc : 0.0
r2 :  -0.0833765078750015
[235. 150. 124.  78. 168. 253.  97. 102. 249. 142.] 의 예측 결과  [ 91. 220.  91.  90.  90.  91.  53.  90. 220. 200.]

KNeighborsClassifier()
score : 0.0
acc : 0.0
r2 :  -0.5564141962639386
[235. 150. 124.  78. 168. 253.  97. 102. 249. 142.] 의 예측 결과  [ 91. 129.  77.  64.  60.  63.  42.  53.  67.  74.]

RandomForestClassifier()
score : 0.02247191011235955
acc : 0.02247191011235955
r2 :  -0.06879755740443882
[235. 150. 124.  78. 168. 253.  97. 102. 249. 142.] 의 예측 결과  [ 91. 232. 156.  95.  53. 138.  83. 214.  67. 158.]

KNeighborsRegressor() - ValueError : Classification metrics can't handle a mix of multiclass and continuous targets
score : 0.38626977834604637
r2 :  0.38626977834604637
[235. 150. 124.  78. 168. 253.  97. 102. 249. 142.] 의 예측 결과  [166.4 190.6 124.  132.6 120.4 120.2 101.4 108.6 145.4 131.6]

RandomForestRegressor() - ValueError : Classification metrics can't handle a mix of multiclass and continuous targets
score : 0.3562699148822518
r2 :  0.3562699148822518
[235. 150. 124.  78. 168. 253.  97. 102. 249. 142.] 의 예측 결과  [171.77 202.01 158.94 124.15 100.09 122.06  94.71 141.59 148.2 
 117.23]
'''