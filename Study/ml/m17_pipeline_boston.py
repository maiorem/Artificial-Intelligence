import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
# from sklearn.utils.testing import all_estimators
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
import warnings
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler


x, y=load_boston(return_X_y=True)

x_train, x_test, y_train, y_test=train_test_split(x, y, train_size=0.8)


parameters= [
    {'jin__n_estimators' : [100,200],
    'jin__max_depth' : [6,8,10,12],
    'jin__max_features' : [2, 5],
    'jin__min_samples_leaf' : [3,5,7,10],
    'jin__min_samples_split' : [2,3,5,10],
    'jin__n_jobs' : [-1]}
] 

#2. 모델
pipe=Pipeline([("scaler", MinMaxScaler()), ('jin', RandomForestRegressor())])
# pipe=make_pipeline(MinMaxScaler(), RandomForestRegressor()) 
model=RandomizedSearchCV(pipe, parameters, cv=5) 


#3. 훈련
model.fit(x_train, y_train) 

#4. 평가, 예측
print("최적의 매개변수 : ", model.best_estimator_)
y_predict=model.predict(x_test)
print('최종정답률 : ', r2_score(y_test, y_predict))

'''
최적의 매개변수 :  Pipeline(steps=[('scaler', MinMaxScaler()),
                ('jin',
                 RandomForestRegressor(max_depth=6, max_features=5,
                                       min_samples_leaf=3, min_samples_split=3,
                                       n_estimators=200, n_jobs=-1))])
최종정답률 :  0.8438513154314851
'''