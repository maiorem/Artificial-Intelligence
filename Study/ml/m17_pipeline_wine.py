import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, r2_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

wine=pd.read_csv('./data/csv/winequality-white.csv', sep=';', header=0)
y=wine['quality']
x=wine.drop('quality', axis=1)

x_train, x_test, y_train, y_test=train_test_split(x, y, train_size=0.8)

parameters= [
    {'jin__n_estimators' : [100,150,200,300],
    'jin__max_depth' : [6,8,10,12,15],
    'jin__min_samples_leaf' : [3,5,7,10,15,20],
    'jin__min_samples_split' : [2,3,5,10,13],
    'jin__n_jobs' : [-1],
    'jin__max_features' : ['auto', 'sqrt', 'log2']}
] 

#2. 모델
pipe=Pipeline([("scaler", MinMaxScaler()), ('jin', RandomForestClassifier())])
# pipe=make_pipeline(MinMaxScaler(), SVC()) 
model=RandomizedSearchCV(pipe, parameters, cv=10, verbose=2)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가
print('acc : ', model.score(x_test, y_test))
print("최적의 매개변수 : ", model.best_estimator_)

'''
acc :  0.6816326530612244
최적의 매개변수 :  Pipeline(steps=[('scaler', MinMaxScaler()),
                ('jin',
                 RandomForestClassifier(max_depth=15, min_samples_leaf=3,
                                        min_samples_split=3, n_estimators=200,
                                        n_jobs=-1))])
'''