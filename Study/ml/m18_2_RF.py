#RandomForest는 Tree들이 모여 형성된 모델.
#Tree구조의 모델들 성능이 다른 모델에 비해 좋음. keras보다 잘 나올 수도 있음.
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer=load_breast_cancer()
x=cancer.data
y=cancer.target

x_train, x_test, y_train, y_test=train_test_split(x, y, train_size=0.8)

# model=DecisionTreeClassifier(max_depth=4)
model=RandomForestClassifier(max_depth=4)
model.fit(x_train, y_train)

acc=model.score(x_test, y_test)

print("acc : ", acc)
print(model.feature_importances_) #컬럼 중요도
'''
acc :  0.9824561403508771
[0.04249221 0.01881232 0.04003434 0.02413542 0.00559851 0.00562279
 0.02633257 0.12884412 0.00248669 0.00362725 0.01588847 0.00248599
 0.01123995 0.02723033 0.00183238 0.00278832 0.00642689 0.00134325
 0.00260337 0.00341519 0.12580872 0.01541016 0.14497406 0.14271002
 0.01274588 0.01998208 0.02898057 0.12347107 0.00537139 0.00730569]
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