# 과적합 방지
# 1. 훈련 데이터량을 늘린다.
# 2. 피쳐 수를 늘린다.
# 3. Regularization

'''
max_depth : 트리의 최대 깊이
learning_rate : 학습 속도
n_estimators : 트리의 갯수
n_jobs : XGBoost를 실행하기 위한 병렬처리(쓰레드) 갯수
colsample_bylevel : 트리 레벨별로 훈련데이터의 변수를 샘플링해주는 비율
colsample_bytree : 각 트리마다의 feature 샘플링 비율
'''
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor, plot_importance

x, y=load_boston(return_X_y=True)

x_train, x_test, y_train, y_test=train_test_split(x, y, train_size=0.9)

model=XGBRegressor(n_estimators=300, 
                    learning_rate=1, 
                    colsample_bytree=1, 
                    colsample_bylevel=1,
                    max_depth=5, 
                    n_jobs=-1)

model.fit(x_train, y_train)

y_predict=model.predict(x_test)
score=model.score(x_test, y_test)
r2=r2_score(y_test, y_predict)
print('score :', score)
print('r2 :', r2)

# XGBoost
# r2 : 0.8916325276205544

# Tensorflow
# r2 :  0.8877071672091482
