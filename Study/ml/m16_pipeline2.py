# PIPE LINE

import pandas as pd
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

warnings.filterwarnings('ignore')

#1. 데이터
iris=pd.read_csv('./data/csv/iris_ys.csv', header=0, index_col=0)

x=iris.iloc[:, :-1] #(150,4)
y=iris.iloc[:, -1] #(150,)

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=66)

parameters=[
    {"jin__C" : [1, 10, 100, 1000], "jin__kernel":["linear"]},
    {"jin__C" : [1, 10, 100, 1000], "jin__kernel":["rbf"], "jin__gamma":[0.001, 0.0001]},
    {"jin__C" : [1, 10, 100, 1000], "jin__kernel":["sigmoid"], "jin__gamma":[0.001, 0.0001]}
] # // Pipeline 파라미터 이름 명시

# parameters=[
#     {"svc__C" : [1, 10, 100, 1000], "svc__kernel":["linear"]},
#     {"svc__C" : [1, 10, 100, 1000], "svc__kernel":["rbf"], "svc__gamma":[0.001, 0.0001]},
#     {"svc__C" : [1, 10, 100, 1000], "svc__kernel":["sigmoid"], "svc__gamma":[0.001, 0.0001]}
# ] // make_pipeline()

# parameters=[
#     {"C" : [1, 10, 100, 1000], "kernel":["linear"]},
#     {"C" : [1, 10, 100, 1000], "kernel":["rbf"], "gamma":[0.001, 0.0001]},
#     {"C" : [1, 10, 100, 1000], "kernel":["sigmoid"], "gamma":[0.001, 0.0001]}
# ] // pipeline을 안 쓴 경우

#2. 모델
pipe=Pipeline([("scaler", MinMaxScaler()), ('jin', SVC())])
# pipe=make_pipeline(MinMaxScaler(), SVC()) 
model=RandomizedSearchCV(pipe, parameters, cv=5)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가
print('acc : ', model.score(x_test, y_test))
print("최적의 매개변수 : ", model.best_estimator_)
# print("최적의 매개변수 : ", model.best_params_)

'''
acc :  0.9666666666666667
최적의 매개변수 :  Pipeline(steps=[('scaler', MinMaxScaler()), ('jin', SVC(C=1, kernel='linear'))])
'''
