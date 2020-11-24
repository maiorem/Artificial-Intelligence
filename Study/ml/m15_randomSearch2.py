# 유방암 데이터
# 모델 : RandomForestClssifier
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
# from sklearn.utils.testing import all_estimators
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
import warnings
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score

#1. 데이터
dataset=load_breast_cancer()
x=dataset.data
y=dataset.target

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=66)

parameters= [
    {'n_estimators' : [100,200],
    'max_depth' : [6,8,10,12],
    'min_samples_leaf' : [3,5,7,10],
    'min_samples_split' : [2,3,5,10],
    'n_jobs' : [-1]}
] #128번

#2. 모델
kfold=KFold(n_splits=3, shuffle=True) # 5번
model=RandomizedSearchCV(RandomForestClassifier(), parameters, cv=kfold, verbose=2) # 총 640번 훈련


#3. 훈련
model.fit(x_train, y_train) 

#4. 평가, 예측
print("최적의 매개변수 : ", model.best_estimator_)
y_predict=model.predict(x_test)
print('최종정답률 : ', accuracy_score(y_test, y_predict))

'''
RandomForestClassifier parameter =>
n_estimators : 생성할 의사결정 나무 개수 
max_features : 의사결정나무 만들 시, 사용하는 feature 개수, default는 auto. 전체 feature 갯수의 제곱근.
min_samples_split : 노드를 분할하기 위한 최소한의 샘플 데이터수
min_samples_leaf : 리프노드가 되기 위해 필요한 최소한의 샘플 데이터수
max_depth : 트리의 최대 깊이
max_leaf_nodes : 리프노드의 최대 개수

GridSearchCV
n_splits=5 => 640번 훈련
최적의 매개변수 :  RandomForestClassifier(max_depth=6, min_samples_leaf=3, n_jobs=-1)
최종정답률 :  0.9649122807017544

RandomizedSearhCV
최적의 매개변수 :  RandomForestClassifier(max_depth=6, min_samples_leaf=3, min_samples_split=10,
                       n_estimators=200, n_jobs=-1)
최종정답률 :  0.9649122807017544
'''