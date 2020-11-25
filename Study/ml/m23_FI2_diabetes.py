# 기준 : XGbost
# 1. feature importance 0 제거
# 2. 하위 30% 제거
# 3. 디폴트와 성능 비교
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import r2_score, accuracy_score
from sklearn.decomposition import PCA

dataset=load_diabetes()
x=dataset.data
y=dataset.target

# pca1=PCA(n_components=7)
# x=pca1.fit_transform(x)

x_train, x_test, y_train, y_test=train_test_split(x, y, train_size=0.8)

model=XGBRegressor(max_depth=4)
model.fit(x_train, y_train)

y_predict=model.predict(x_test)

print("score :", r2_score(y_test, y_predict))
print(model.feature_importances_)

'''
Default
score : 0.43378623375119496
[0.02785363 0.08743825 0.15927461 0.07139384 0.0495851  0.05730768
 0.05833183 0.05959074 0.37548473 0.05373957]
 
30% 제거
score : 0.31112262069744623
[0.28740177 0.09944782 0.13956098 0.23825209 0.06044487 0.0691275
 0.10576495]

# '''