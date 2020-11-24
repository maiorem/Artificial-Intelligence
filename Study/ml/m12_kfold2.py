# 4개의 모델을 완성

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

# model=SVC()
# model=LinearSVC()
# model=KNeighborsClassifier()
model=RandomForestClassifier()

scores=cross_val_score(model, x_train, y_train, cv=kfold)

print('scores : ', scores)

'''
SVC()
scores :  [0.95833333 1.         0.95833333 1.         1.        ]

LinearSVC()
scores :  [0.79166667 0.83333333 0.875      0.83333333 0.83333333]

KNeighborsClassifier()
scores :  [1.         1.         1.         0.95833333 1.        ]

RandomForestClassifier()
scores :  [1. 1. 1. 1. 1.]
'''