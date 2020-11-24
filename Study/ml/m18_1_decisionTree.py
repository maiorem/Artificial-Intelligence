#RandomForest는 Tree들이 모여 형성된 모델.
#Tree구조의 모델들 성능이 다른 모델에 비해 좋음. keras보다 잘 나올 수도 있음.
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer=load_breast_cancer()
x=cancer.data
y=cancer.target

x_train, x_test, y_train, y_test=train_test_split(x, y, train_size=0.8)

model=DecisionTreeClassifier(max_depth=4)
model.fit(x_train, y_train)

acc=model.score(x_test, y_test)

print("acc : ", acc)
print(model.feature_importances_) #컬럼 중요도
'''
acc :  0.9385964912280702
[0.         0.03108845 0.         0.00902341 0.         0.
 0.02456371 0.         0.         0.         0.         0.
 0.         0.         0.00838895 0.         0.         0.
 0.         0.         0.72821954 0.03986021 0.         0.
 0.         0.         0.         0.14945008 0.00940566 0.        ]
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