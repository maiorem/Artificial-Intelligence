#RandomForest는 Tree들이 모여 형성된 모델.
#Tree구조의 모델들 성능이 다른 모델에 비해 좋음. keras보다 잘 나올 수도 있음.
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor, GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

cancer=load_breast_cancer()
x=cancer.data
y=cancer.target
print(x.shape)

x_train, x_test, y_train, y_test=train_test_split(x, y, train_size=0.8)

# model=DecisionTreeClassifier(max_depth=4)
# model=RandomForestClassifier(max_depth=4)
# model=GradientBoostingClassifier(max_depth=4)
model=XGBClassifier(max_depth=4)
model.fit(x_train, y_train)

acc=model.score(x_test, y_test)

print("acc : ", acc)
print(model.feature_importances_) #컬럼 중요도
'''
acc :  0.9824561403508771
[0.         0.01780334 0.         0.01607641 0.00450007 0.
 0.00979987 0.04703457 0.00056042 0.         0.02001974 0.00698099
 0.01122535 0.00486326 0.00154637 0.00221109 0.00178483 0.00728818
 0.002151   0.00240713 0.5314688  0.01168262 0.17006911 0.05242056
 0.0054902  0.0050082  0.0054588  0.05925446 0.00193505 0.00095945]
'''

import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importances_cancer(model) :
    n_features=cancer.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)

plot_feature_importances_cancer(model)
plt.show()