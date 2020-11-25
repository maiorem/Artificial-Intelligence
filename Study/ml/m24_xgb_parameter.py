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

from xgboost import XGBClassifier, XGBRegressor, plot_importance

model=XGBRegressor()