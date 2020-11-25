# 기준 : XGbost
# 1. feature importance 0 제거
# 2. 하위 30% 제거
# 3. 디폴트와 성능 비교
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA

dataset=load_boston()
x=dataset.data
y=dataset.target

# pca1=PCA(n_components=9)
# x=pca1.fit_transform(x)

x_train, x_test, y_train, y_test=train_test_split(x, y, train_size=0.8)

model=XGBRegressor(max_depth=4)
model.fit(x_train, y_train)

y_predict=model.predict(x_test)

print("R2 :", r2_score(y_test, y_predict))
print(model.feature_importances_)

'''
Default
R2 : 0.9090415131940273
[0.02130319 0.00286287 0.01434059 0.01714036 0.0471935  0.15154535
 0.0130349  0.06681163 0.0114954  0.0272664  0.04258548 0.01972253
 0.5646978 ]


30% 제거
R2 : 0.6682821217055774
[0.15831138 0.02710617 0.15774179 0.02702889 0.13498697 0.35119942
 0.03275908 0.04039648 0.07046985]
# '''