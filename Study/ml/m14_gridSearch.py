# Grid Search : 하이퍼 파라미터를 싹 모아서 체로 걸러 fit

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
# from sklearn.utils.testing import all_estimators
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
import warnings
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score

warnings.filterwarnings('ignore')

#1. 데이터
iris=pd.read_csv('./data/csv/iris_ys.csv', header=0, index_col=0)

x=iris.iloc[:, :-1]
y=iris.iloc[:, -1]

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=66)

parameters=[
    {"C" : [1, 10, 100, 1000], "kernel":["linear"]},
    {"C" : [1, 10, 100, 1000], "kernel":["rbf"], "gamma":[0.001, 0.0001]},
    {"C" : [1, 10, 100, 1000], "kernel":["sigmoid"], "gamma":[0.001, 0.0001]}
] # 20번 (4+8+8)

#2. 모델
kfold=KFold(n_splits=5, shuffle=True) # 5번

#model=SVC()
model=GridSearchCV(SVC(), parameters, cv=kfold) # 총 100번 훈련


#3. 훈련
model.fit(x_train, y_train) 

#4. 평가, 예측
print("최적의 매개변수 : ", model.best_estimator_)
y_predict=model.predict(x_test)
print('최종정답률 : ', accuracy_score(y_test, y_predict))

'''
최적의 매개변수 :  SVC(C=1, kernel='linear')
최종정답률 :  0.9666666666666667
'''