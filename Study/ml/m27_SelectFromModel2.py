# 실습
# 1. 상단 모델에 그리드서치 또는 랜덤 서치 적용
# 최적의 R2값과 feature importance 구할 것

# 2. 위 쓰레드값으로 SelectFromModel을 구해서 최적의 feature 갯수를 구할 것

# 3. 위 feature 갯수로 데이터 컬럼을 수정(삭제)해서 그리드서치 또는 랜덤서치 적용하고
# 최적의 R2값을 구할 것

# 1번과 2번 비교해 볼 것!!!!

import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier, XGBRegressor, plot_importance
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel

# # 1.
# x, y=load_boston(return_X_y=True)

# x_train, x_test, y_train, y_test=train_test_split(x, y, train_size=0.8)


# parameters= [
#     {'n_estimators' : [100,200, 300],
#     'learning_rate' : [0.1,0.3,0.001,0.01],
#     'max_depth' : [4,5,6]}, 
#     {'n_estimators' : [100,200, 300],
#     'learning_rate' : [0.1, 0.001, 0.01],
#     'max_depth' : [4,5,6],
#     'colsample_bytree' :[0.6, 0.9, 1]},
#     {'n_estimators' : [90, 110],
#     'learning_rate' : [0.1, 0.001, 0.5],
#     'max_depth' : [4,5,6],
#     'colsample_bytree' :[0.6, 0.9, 1],
#     'colsample_bylevel' :[0.6, 0.7, 0.9]}
# ] 

# #2. 모델
# kfold=KFold(n_splits=5, shuffle=True)
# model=RandomizedSearchCV(XGBRegressor(), parameters, cv=kfold) 


# #3. 훈련
# model.fit(x_train, y_train) 

# #4. 평가, 예측
# print("최적의 매개변수 : ", model.best_estimator_)
# y_predict=model.predict(x_test)
# print('최종정답률 : ', r2_score(y_test, y_predict))
# print('feature_importance : ', XGBRegressor().fit(x_train, y_train).feature_importances_)


# '''
# 최적의 매개변수 :  XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
#              importance_type='gain', interaction_constraints='',
#              learning_rate=0.1, max_delta_step=0, max_depth=6,
#              min_child_weight=1, missing=nan, monotone_constraints='()',
#              n_estimators=200, n_jobs=0, num_parallel_tree=1, random_state=0,
#              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
#              tree_method='exact', validate_parameters=1, verbosity=None)
# 최종정답률 :  0.9164073955314002
# feature_importance :  [0.01298503 0.00230451 0.00792385 0.00134126 0.03422489 0.23503631
#  0.0154811  0.07551808 0.00689286 0.03824459 0.02927438 0.0052571
#  0.53551614]
# '''

# # 2.
# x, y=load_boston(return_X_y=True)

# x_train, x_test, y_train, y_test=train_test_split(x, y, train_size=0.8)

# model=XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
#              importance_type='gain', interaction_constraints='',
#              learning_rate=0.1, max_delta_step=0, max_depth=6,
#              min_child_weight=1, monotone_constraints='()',
#              n_estimators=200, n_jobs=0, num_parallel_tree=1, random_state=0,
#              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
#              tree_method='exact', validate_parameters=1, verbosity=None)

# model.fit(x_train, y_train)
# score=model.score(x_test, y_test)

# print('score :', score)

# thresholds=np.sort(model.feature_importances_)
# print('feature_importance SORT : ',thresholds)

# for thresh in thresholds :
#     selection=SelectFromModel(model, threshold=thresh, prefit=True)
    
#     select_x_train=selection.transform(x_train)
    
#     selection_model=XGBRegressor(n_jobs=-1)
#     selection_model.fit(select_x_train, y_train)

#     selec_x_test=selection.transform(x_test)
#     y_predict=selection_model.predict(selec_x_test)

#     score=r2_score(y_test, y_predict)

#     print("Thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1], score*100.0))


# '''
# score : 0.899114154321254
# feature_importance SORT :  [0.0025913  0.0027841  0.00816071 0.01063932 0.01107433 0.01394466
#  0.0190807  0.03944993 0.04366116 0.05075784 0.0737866  0.20424435
#  0.5198251 ]
# Thresh=0.003, n=13, R2: 91.83%
# Thresh=0.003, n=12, R2: 92.00%
# Thresh=0.008, n=11, R2: 91.61%
# Thresh=0.011, n=10, R2: 90.86%
# Thresh=0.011, n=9, R2: 91.23%
# Thresh=0.014, n=8, R2: 91.77%
# Thresh=0.019, n=7, R2: 91.39%
# Thresh=0.039, n=6, R2: 88.57%
# Thresh=0.044, n=5, R2: 89.24%
# Thresh=0.051, n=4, R2: 88.70%
# Thresh=0.074, n=3, R2: 90.21%
# Thresh=0.204, n=2, R2: 80.71%
# Thresh=0.520, n=1, R2: 58.04%
# '''


# 3.
x, y=load_boston(return_X_y=True)
x_data1=x[:,:4]
x_data2=x[:,5:]
x=np.concatenate([x_data1, x_data2], axis=1)
print(x.shape) #(506, 12)

x_train, x_test, y_train, y_test=train_test_split(x, y, train_size=0.8)

parameters= [
    {'n_estimators' : [100,200, 300],
    'learning_rate' : [0.1,0.3,0.001,0.01],
    'max_depth' : [4,5,6]}, 
    {'n_estimators' : [100,200, 300],
    'learning_rate' : [0.1, 0.001, 0.01],
    'max_depth' : [4,5,6],
    'colsample_bytree' :[0.6, 0.9, 1]},
    {'n_estimators' : [90, 110],
    'learning_rate' : [0.1, 0.001, 0.5],
    'max_depth' : [4,5,6],
    'colsample_bytree' :[0.6, 0.9, 1],
    'colsample_bylevel' :[0.6, 0.7, 0.9]}
] 

#2. 모델
kfold=KFold(n_splits=5, shuffle=True)
model=RandomizedSearchCV(XGBRegressor(), parameters, cv=kfold) 


#3. 훈련
model.fit(x_train, y_train) 

#4. 평가, 예측
print("최적의 매개변수 : ", model.best_estimator_)
y_predict=model.predict(x_test)
print('최종정답률 : ', r2_score(y_test, y_predict))
print('feature_importance : ', XGBRegressor().fit(x_train, y_train).feature_importances_)

'''
최적의 매개변수 :  XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=0.6,
             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.1, max_delta_step=0, max_depth=4,
             min_child_weight=1, missing=nan, monotone_constraints='()',
             n_estimators=110, n_jobs=0, num_parallel_tree=1, random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
             tree_method='exact', validate_parameters=1, verbosity=None)
최종정답률 :  0.9374078526207467
feature_importance :  [0.03148203 0.00188427 0.01199806 0.00124609 0.23142023 0.01452872
 0.0400641  0.01199174 0.02808359 0.03617088 0.01023689 0.58089346]
'''