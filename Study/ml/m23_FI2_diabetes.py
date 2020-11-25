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


# #PCA로 컬럼 걸러내기
# pca=PCA()
# pca.fit(x)
# cumsum=np.cumsum(pca.explained_variance_ratio_) #누적된 합 표시
# # print(cumsum)

# d=np.argmax(cumsum >= 0.8) + 1
# # print(cumsum>=0.95) 
# print(d) # 2 4

pca1=PCA(n_components=7)
x=pca1.fit_transform(x)

x_train, x_test, y_train, y_test=train_test_split(x, y, train_size=0.8)

model=XGBClassifier(max_depth=4)
model.fit(x_train, y_train)

y_predict=model.predict(x_test)

print("score :", r2_score(y_test, y_predict))
print(model.feature_importances_)

'''
Default
score : 0.02247191011235955
[0.11618966 0.10648346 0.10108944 0.1020179  0.08192872 0.08790331
 0.10599291 0.10425983 0.09501596 0.09911878]

30% 제거
score : -0.012755094898871722
[0.14898889 0.14885664 0.13235529 0.15608892 0.13421012 0.13919918
 0.14030096]

# '''