# PIPE LINE

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
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

warnings.filterwarnings('ignore')

#1. 데이터
iris=pd.read_csv('./data/csv/iris_ys.csv', header=0, index_col=0)

x=iris.iloc[:, :-1]
y=iris.iloc[:, -1]

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=66)

pipe=Pipeline([("scaler", MinMaxScaler()), ('svm', SVC())])
# pipe=make_pipeline(MinMaxScaler(), SVC()) #cv를 하면 validation만 중복 scaling을 하게 되어 과적합에 걸리므로 cv는 하지 않는다.
pipe.fit(x_train, y_train)

print('acc : ', pipe.score(x_test, y_test))
# acc :  1.0


