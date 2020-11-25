# 기준 : XGbost
# 1. feature importance 0 제거
# 2. 하위 30% 제거
# 3. 디폴트와 성능 비교

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

iris=load_iris()
x=iris.data
y=iris.target

x=x[:,1:]

x_train, x_test, y_train, y_test=train_test_split(x, y, train_size=0.8)

model=XGBClassifier(max_depth=4)
model.fit(x_train, y_train)

acc=model.score(x_test, y_test)

print("acc : ", acc)
print(model.feature_importances_)

'''
Default
acc :  0.9333333333333333
[0.00831502 0.02227198 0.53397155 0.4354414 ]

30% 제거
acc :  0.9333333333333333
[0.02230631 0.6154553  0.36223838]

'''