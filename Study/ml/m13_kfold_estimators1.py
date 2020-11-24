# 분류

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils.testing import all_estimators
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import warnings


warnings.filterwarnings('ignore')

iris=pd.read_csv('./data/csv/iris_ys.csv', header=0, index_col=0)

x=iris.iloc[:, 0:4]
y=iris.iloc[:, 4]

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=66)

allAlgorithms=all_estimators(type_filter='classifier')
kfold=KFold(n_splits=10, shuffle=True)

for (name, algorithm) in allAlgorithms :
    try :
        model=algorithm()
        scores=cross_val_score(model, x_train, y_train, cv=kfold)
        print(name, '의 정답률 : \n', scores)
    except :
        pass


'''
AdaBoostClassifier 의 정답률 :  
[0.83333333 1.         0.83333333 0.83333333 0.58333333 0.91666667 0.83333333 1.         0.91666667 0.75      ]
BaggingClassifier 의 정답률 :  
[1.         0.83333333 0.91666667 1.         1.         0.91666667 1.         0.83333333 1.         1.        ]
BernoulliNB 의 정답률 :  
[0.25       0.33333333 0.08333333 0.08333333 0.25       0.33333333 0.16666667 0.41666667 0.25       0.25      ]
CalibratedClassifierCV 의 정답률 :  
[0.75       0.83333333 1.         1.         1.         0.91666667 1.         0.83333333 1.         0.75      ]
CategoricalNB 의 정답률 :  
[1.         1.         0.91666667 0.91666667 0.91666667 0.91666667 0.91666667 0.83333333 1.         1.        ]
CheckingClassifier 의 정답률 :  
[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
ComplementNB 의 정답률 :  
[0.66666667 0.58333333 0.75       0.5        0.66666667 0.41666667 1.         0.66666667 0.66666667 0.75      ]
DecisionTreeClassifier 의 정답률 :  
[0.83333333 1.         0.91666667 1.         0.91666667 1. 0.91666667 0.91666667 0.91666667 1.        ]
DummyClassifier 의 정답률 :  
[0.25       0.5        0.41666667 0.25       0.41666667 0.16666667 0.33333333 0.41666667 0.33333333 0.08333333]
ExtraTreeClassifier 의 정답률 :  
[1.         0.83333333 0.83333333 1.         1.         0.91666667 0.91666667 1.         1.         1.        ]
ExtraTreesClassifier 의 정답률 :  
[0.91666667 1.         1.         0.91666667 0.83333333 0.91666667 1.         1.         1.         0.91666667]
GaussianNB 의 정답률 :  
[1.         1.         1.         0.91666667 0.83333333 0.83333333 1.         0.83333333 1.         1.        ]
GaussianProcessClassifier 의 정답률 :  
[1.         1.         1.         1.         1.         0.91666667 1.         0.91666667 1.         0.91666667]
GradientBoostingClassifier 의 정답률 :  
[0.91666667 0.91666667 0.91666667 0.91666667 1.         1. 0.91666667 0.91666667 0.91666667 0.91666667]
HistGradientBoostingClassifier 의 정답률 :  
[1.         1.         0.91666667 0.91666667 1.         0.91666667 1.         0.91666667 0.83333333 0.91666667]
KNeighborsClassifier 의 정답률 :  
[0.91666667 1.         0.91666667 1.         1.         0.91666667 1.         1.         0.91666667 1.        ]
LabelPropagation 의 정답률 :  
[1.         0.91666667 1.         1.         0.91666667 1. 0.91666667 0.83333333 0.91666667 1.        ]
LabelSpreading 의 정답률 :  
[0.91666667 0.91666667 1.         1.         1.         1. 0.91666667 0.91666667 1.         1.        ]
LinearDiscriminantAnalysis 의 정답률 :  
[1.         0.91666667 1.         1.         1.         1. 1.         0.91666667 0.91666667 1.        ]
LinearSVC 의 정답률 :  
[1.         0.91666667 1.         0.91666667 1.         0.91666667 0.91666667 1.         0.91666667 1.        ]
LogisticRegression 의 정답률 :  
[0.83333333 0.91666667 1.         1.         1.         1. 0.91666667 1.         0.83333333 1.        ]
LogisticRegressionCV 의 정답률 :  
[1.         0.91666667 1.         1.         1.         0.75 1.         1.         1.         0.91666667]
MLPClassifier 의 정답률 :  
[1.         0.91666667 1.         1.         1.         1. 0.91666667 1.         1.         1.        ]
MultinomialNB 의 정답률 :  
[0.83333333 1.         0.83333333 1.         0.83333333 1. 0.75       0.75       1.         0.58333333]
NearestCentroid 의 정답률 :  
[0.91666667 1.         1.         0.91666667 0.66666667 1.
NuSVC 의 정답률 :  [0.91666667 1.         1.         1.         0.91666667 1.
 0.91666667 1.         0.91666667 1.        ]
PassiveAggressiveClassifier 의 정답률 :  [1.         0.66666667 1.         1.         0.83333333 0.91666667
 1.         0.58333333 0.66666667 1.        ]
Perceptron 의 정답률 :  [0.91666667 0.83333333 0.5        0.58333333 0.91666667 0.91666667
 0.66666667 0.33333333 0.58333333 0.66666667]
QuadraticDiscriminantAnalysis 의 정답률 :  [1.         1.         1.         1.         0.91666667 0.91666667
 1.         0.91666667 1.         0.83333333]
RandomForestClassifier 의 정답률 :  [1.         1.         0.83333333 1.         1.         0.83333333
 0.91666667 1.         0.91666667 0.91666667]
RidgeClassifier 의 정답률 :  [0.83333333 0.83333333 1.         0.91666667 0.66666667 1.
 0.75       0.91666667 0.83333333 0.83333333]
RidgeClassifierCV 의 정답률 :  [1.         0.83333333 0.66666667 0.75       0.91666667 0.83333333
 0.83333333 0.75       0.83333333 0.83333333]
SGDClassifier 의 정답률 :  [0.83333333 0.91666667 0.66666667 1.         0.75       0.5
 0.83333333 0.91666667 0.75       0.75      ]
SVC 의 정답률 :  [1.         0.91666667 1.         0.91666667 0.91666667 1.
 1.         1.         1.         0.83333333]
'''