# 실습
# 1. 상단 모델에 그리드서치 또는 랜덤 서치 적용
# 최적의 R2값과 feature importance 구할 것

# 2. 위 쓰레드값으로 SelectFromModel을 구해서 최적의 feature 갯수를 구할 것

# 3. 위 feature 갯수로 데이터 컬럼을 수정(삭제)해서 그리드서치 또는 랜덤서치 적용하고
# 최적의 R2값을 구할 것

# 1번과 2번 비교해 볼 것!!!!

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier, XGBRegressor, plot_importance
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel

# # 1.
# x, y=load_diabetes(return_X_y=True)

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
#              colsample_bynode=1, colsample_bytree=0.9, gamma=0, gpu_id=-1,
#              importance_type='gain', interaction_constraints='',
#              learning_rate=0.01, max_delta_step=0, max_depth=4,
#              min_child_weight=1, missing=nan, monotone_constraints='()',
#              n_estimators=300, n_jobs=0, num_parallel_tree=1, random_state=0,
#              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
#              tree_method='exact', validate_parameters=1, verbosity=None)
# 최종정답률 :  0.48322837581769684
# feature_importance :  [0.03662913 0.06922864 0.15622318 0.07285699 0.03576181 0.05462984
#  0.0438249  0.07072538 0.39779556 0.0623245 ]
# '''

# 2.
x, y=load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test=train_test_split(x, y, train_size=0.8)

model=XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=0.9, gamma=0, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.01, max_delta_step=0, max_depth=4,
             min_child_weight=1, monotone_constraints='()',
             n_estimators=300, n_jobs=0, num_parallel_tree=1, random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
             tree_method='exact', validate_parameters=1, verbosity=None)

model.fit(x_train, y_train)
score=model.score(x_test, y_test)

print('score :', score)

thresholds=np.sort(model.feature_importances_)
print('feature_importance SORT : ',thresholds)

for thresh in thresholds :
    selection=SelectFromModel(model, threshold=thresh, prefit=True)
    
    select_x_train=selection.transform(x_train)
    
    selection_model=XGBRegressor(n_jobs=-1)
    selection_model.fit(select_x_train, y_train)

    selec_x_test=selection.transform(x_test)
    y_predict=selection_model.predict(selec_x_test)

    score=r2_score(y_test, y_predict)

    print("Thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1], score*100.0))


'''
score : 0.44242720157272164
feature_importance SORT :  [0.02910201 0.04078781 0.04445104 0.0521705  0.06195119 0.08438916
 0.09157841 0.0975055  0.22566383 0.2724005 ]
Thresh=0.029, n=10, R2: 39.29%
Thresh=0.041, n=9, R2: 37.33%
Thresh=0.044, n=8, R2: 46.17%
Thresh=0.052, n=7, R2: 44.13%
Thresh=0.062, n=6, R2: 43.32%
Thresh=0.084, n=5, R2: 36.03%
Thresh=0.092, n=4, R2: 33.32%
Thresh=0.098, n=3, R2: 22.84%
Thresh=0.226, n=2, R2: 14.58%
Thresh=0.272, n=1, R2: -4.88%
'''


# # 3.
# x, y=load_diabetes(return_X_y=True)
# x_data1=x[:,1:5]
# x_data2=x[:,6:]
# x=np.concatenate([x_data1, x_data2], axis=1)
# print(x.shape) #(442, 8)

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
# 최적의 매개변수 :  XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=0.6,
#              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
#              importance_type='gain', interaction_constraints='',
#              learning_rate=0.1, max_delta_step=0, max_depth=4,
#              min_child_weight=1, missing=nan, monotone_constraints='()',
#              n_estimators=110, n_jobs=0, num_parallel_tree=1, random_state=0,
#              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
#              tree_method='exact', validate_parameters=1, verbosity=None)
# 최종정답률 :  0.9374078526207467
# feature_importance :  [0.03148203 0.00188427 0.01199806 0.00124609 0.23142023 0.01452872
#  0.0400641  0.01199174 0.02808359 0.03617088 0.01023689 0.58089346]
# '''