import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA

wine=pd.read_csv('./data/csv/winequality-white.csv', sep=';', header=0)
y=wine['quality']
x=wine.drop('quality', axis=1)

print(x.shape) #(4898, 11)
print(y.shape) #(4898,)


# pca1=PCA(n_components=7)
# x=pca1.fit_transform(x)


x_train, x_test, y_train, y_test=train_test_split(x, y, train_size=0.8)


model=XGBClassifier(max_depth=4)
model.fit(x_train, y_train)

y_predict=model.predict(x_test)

print("acc :", model.score(x_test, y_test))
print(model.feature_importances_)

'''
Default
acc : 0.6163265306122448
[0.06848848 0.11636396 0.07834077 0.0771647  0.06426705 0.09034036
 0.06354367 0.06986458 0.06286406 0.06478106 0.24398136]

30% 제거
acc : 0.6244897959183674
[0.13303144 0.14161955 0.1353884  0.23206569 0.11359904 0.11410286
 0.13019301]
# '''