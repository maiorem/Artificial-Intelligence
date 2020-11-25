# 기준 : XGbost
# 1. feature importance 0 제거
# 2. 하위 30% 제거
# 3. 디폴트와 성능 비교
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import r2_score, accuracy_score
from sklearn.decomposition import PCA

dataset=load_diabetes()
x=dataset.data
y=dataset.target

pca1=PCA(n_components=7)
x=pca1.fit_transform(x)

x_train, x_test, y_train, y_test=train_test_split(x, y, train_size=0.8)

model=XGBClassifier(max_depth=4)
model.fit(x_train, y_train)

y_predict=model.predict(x_test)

print("score :", model.score(x_test, y_test))
print(model.feature_importances_)

'''
Default
score : 0.0
[0.09753311 0.10002925 0.10379826 0.10192552 0.08558202 0.08127745
 0.11422036 0.11153243 0.10929769 0.09480391]

30% 제거
score : 0.02247191011235955
[0.17569584 0.1307726  0.14014727 0.15188344 0.13347404 0.13590014
 0.1321266 ]

# '''