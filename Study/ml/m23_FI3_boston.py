# 기준 : XGbost
# 1. feature importance 0 제거
# 2. 하위 30% 제거
# 3. 디폴트와 성능 비교
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA

dataset=load_boston()
x=dataset.data
y=dataset.target

pca1=PCA(n_components=12)
x=pca1.fit_transform(x)

x_train, x_test, y_train, y_test=train_test_split(x, y, train_size=0.8)

model=XGBClassifier(max_depth=4)
model.fit(x_train, y_train)

y_predict=model.predict(x_test)

print("R2 :", r2_score(y_test, y_predict))
print(model.feature_importances_)

'''
Default
R2 : 0.6321515538653386
[0.08557149 0.08373747 0.08950646 0.         0.09024695 0.08718417
 0.08103242 0.08615365 0.07397702 0.08104043 0.07738417 0.06890872
 0.09525705]

0 제거
R2 : 0.7853844867936511
[0.09855166 0.08328348 0.07799172 0.07547305 0.08213176 0.0908768
 0.08680062 0.08512171 0.07622263 0.07961532 0.08428118 0.07965012]
 
30% 제거
R2 : 0.606120909321445
[0.12426724 0.10687196 0.11397473 0.10534319 0.1044957  0.11582578
 0.11079461 0.11582579 0.10260104]
# '''