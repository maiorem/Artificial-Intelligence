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
        print(name, '의 정답률 : ', scores)
    except :
        pass


'''
AdaBoostClassifier 의 정답률 :  [0.91666667 0.91666667 0.95833333 0.875      0.79166667]
BaggingClassifier 의 정답률 :  [0.91666667 0.95833333 0.95833333 0.95833333 0.875     ]
BernoulliNB 의 정답률 :  [0.29166667 0.33333333 0.29166667 0.29166667 0.375     ]
CalibratedClassifierCV 의 정답률 :  [0.79166667 0.91666667 0.95833333 0.95833333 0.91666667]
CategoricalNB 의 정답률 :  [1.         0.875      0.95833333 0.95833333 0.91666667]
CheckingClassifier 의 정답률 :  [0. 0. 0. 0. 0.]
ComplementNB 의 정답률 :  [0.70833333 0.75       0.58333333 0.70833333 0.58333333]
DecisionTreeClassifier 의 정답률 :  [0.95833333 0.91666667 1.         0.875      0.95833333]
DummyClassifier 의 정답률 :  [0.45833333 0.41666667 0.16666667 0.375      0.33333333]
ExtraTreeClassifier 의 정답률 :  [0.91666667 1.         0.875      1.         0.91666667]
ExtraTreesClassifier 의 정답률 :  [0.91666667 0.95833333 1.         1.         0.875     ]
GaussianNB 의 정답률 :  [0.875      0.91666667 0.95833333 1.         1.        ]
GaussianProcessClassifier 의 정답률 :  [0.95833333 0.91666667 1.         0.91666667 0.95833333]
GradientBoostingClassifier 의 정답률 :  [0.875      1.         0.91666667 0.91666667 0.95833333]
HistGradientBoostingClassifier 의 정답률 :  [1.         0.95833333 0.95833333 0.91666667 0.91666667]
KNeighborsClassifier 의 정답률 :  [1.         0.95833333 1.         0.95833333 0.95833333]
LabelPropagation 의 정답률 :  [0.95833333 0.875      1.         0.95833333 0.95833333]
LabelSpreading 의 정답률 :  [0.95833333 0.95833333 0.95833333 0.95833333 1.        ]
LinearDiscriminantAnalysis 의 정답률 :  [1.         1.         1.         0.95833333 0.91666667]
LinearSVC 의 정답률 :  [1.         0.95833333 0.875      1.         1.        ]
LogisticRegression 의 정답률 :  [0.875      0.95833333 0.95833333 1.         0.95833333]
LogisticRegressionCV 의 정답률 :  [1.         0.91666667 0.79166667 0.95833333 1.        ]
MLPClassifier 의 정답률 :  [1.         1.         0.95833333 0.95833333 0.95833333]
MultinomialNB 의 정답률 :  [0.95833333 0.875      0.875      0.875      0.79166667]
NearestCentroid 의 정답률 :  [0.95833333 1.         0.91666667 0.83333333 0.91666667]
NuSVC 의 정답률 :  [0.95833333 1.         0.95833333 0.875      0.91666667]
PassiveAggressiveClassifier 의 정답률 :  [0.91666667 1.         0.75       0.91666667 0.875     ]
Perceptron 의 정답률 :  [0.83333333 0.75       0.66666667 0.375      0.70833333]
QuadraticDiscriminantAnalysis 의 정답률 :  [1.         1.         0.91666667 0.91666667 1.        ]
RadiusNeighborsClassifier 의 정답률 :  [1.         0.91666667 0.95833333 1.         1.        ]
RandomForestClassifier 의 정답률 :  [0.95833333 0.875      0.95833333 1.         0.875     ]
RidgeClassifier 의 정답률 :  [0.75       0.875      0.95833333 0.79166667 0.83333333]
RidgeClassifierCV 의 정답률 :  [1.         0.70833333 0.875      0.83333333 0.79166667]
SGDClassifier 의 정답률 :  [0.79166667 0.83333333 0.79166667 0.70833333 0.95833333]
SVC 의 정답률 :  [0.95833333 1.         0.91666667 0.95833333 0.95833333]
'''