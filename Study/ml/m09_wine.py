import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

# 1. 데이터
#dataset=load_wine()
#feature_name=dataset.feature_names
#print(feature_name)
#['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']
x, y=load_wine(return_X_y=True)
# print(y) # 분류
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
score : 0.9722222222222222
acc : 0.9722222222222222
[2 1 1 0 1 1 2 0 0 1] 의 예측 결과  [2 1 1 0 1 1 2 0 0 0]

SVC() 
score : 1.0
acc : 1.0
[2 1 1 0 1 1 2 0 0 1] 의 예측 결과  [2 1 1 0 1 1 2 0 0 1]

KNeighborsClassifier()
score : 1.0
acc : 1.0
[2 1 1 0 1 1 2 0 0 1] 의 예측 결과  [2 1 1 0 1 1 2 0 0 1]

RandomForestClassifier()
score : 1.0
acc : 1.0
[2 1 1 0 1 1 2 0 0 1] 의 예측 결과  [2 1 1 0 1 1 2 0 0 1]

KNeighborsRegressor()
ValueError: Classification metrics can't handle a mix of multiclass and continuous targets

RandomForestRegressor() 
ValueError: Classification metrics can't handle a mix of multiclass and continuous targets
'''