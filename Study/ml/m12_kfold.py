import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
#1. 데이터
iris=pd.read_csv('./data/csv/iris_ys.csv', header=0)
x=iris.iloc[:,:4]
y=iris.iloc[:,-1]

print(x.shape, y.shape) #(150, 4) (150,)

x_train, x_test, y_train, y_test=train_test_split(x, y, random_state=66, train_size=0.8)

#2. 모델
kfold=KFold(n_splits=5, shuffle=True)

model=SVC()

scores=cross_val_score(model, x_train, y_train, cv=kfold)

print('scores : ', scores)
#scores :  [0.95833333 1.         0.95833333 1.         1.        ]


# #3. 훈련
# model.fit(x_train, y_train)

# #4, 평가
# # 4. 평가, 예측
# score=model.score(x_test, y_test)

# # accuracy_score를 넣어서 비교할 것
# # 회귀모델일 경우 r2_score와 비교할 것

# y_predict=model.predict(x_test)
# acc=accuracy_score(y_test, y_predict)
# # r2=r2_score(y_test, y_predict)

# print('score :', score)
# print('acc :', acc)
# # print('r2 : ', r2)

# print(y_test[:10], '의 예측 결과 ', y_predict[:10])
