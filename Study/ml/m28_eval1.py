from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score, accuracy_score

#1. 데이터
x, y=load_boston(return_X_y=True)
x_train, x_test, y_train, y_test=train_test_split(x, y, train_size=0.8, random_state=77)

#2. 모델
model=XGBRegressor(n_estimators=1000, learning_rate=0.1)

#3. 훈련
model.fit(x_train, y_train, verbose=True, eval_metric='rmse', eval_set=[(x_test, y_test)])

#4. 평가
score=model.score(x_test, y_test)

print('score :', score)
