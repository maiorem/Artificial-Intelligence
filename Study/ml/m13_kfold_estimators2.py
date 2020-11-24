# 회귀

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils.testing import all_estimators
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import warnings


warnings.filterwarnings('ignore')

iris=pd.read_csv('./data/csv/boston_house_prices.csv', header=1, index_col=None)

x=iris.iloc[:, 0:-1]
y=iris.iloc[:, -1]

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=66)

allAlgorithms=all_estimators(type_filter='regressor')
kfold=KFold(n_splits=5, shuffle=True)

for (name, algorithm) in allAlgorithms :
    try :
        model=algorithm()
        scores=cross_val_score(model, x_train, y_train, cv=kfold)
        print(name, '의 정답률 : ', scores)
    except :
        pass

'''
ARDRegression 의 정답률 :  [0.71339969 0.48583463 0.67814294 0.74282434 0.76605223]
AdaBoostRegressor 의 정답률 :  [0.6139673  0.83124345 0.78530355 0.78441652 0.83699793]
BaggingRegressor 의 정답률 :  [0.66081926 0.84730491 0.86676959 0.83462444 0.88399875]
BayesianRidge 의 정답률 :  [0.61217021 0.66837227 0.75581534 0.74663888 0.59692061]
CCA 의 정답률 :  [0.67588682 0.65880691 0.44940153 0.73855329 0.74625735]
DecisionTreeRegressor 의 정답률 :  [0.66875767 0.36557595 0.75847608 0.67822498 0.64015851]
DummyRegressor 의 정답률 :  [-0.00321731 -0.00350943 -0.00014888 -0.00063683 -0.01495698]
ElasticNet 의 정답률 :  [0.58963495 0.73250267 0.69827191 0.53038184 0.6707371 ]
ElasticNetCV 의 정답률 :  [0.62150728 0.63303509 0.51753561 0.69167953 0.66521012]
ExtraTreeRegressor 의 정답률 :  [0.54297565 0.60597961 0.35541823 0.73310429 0.7085915 ]
ExtraTreesRegressor 의 정답률 :  [0.82127062 0.80617479 0.91219967 0.78521278 0.91215483]
GammaRegressor 의 정답률 :  [-0.02745296 -0.01962056 -0.00133603 -0.01980166 -0.0048519 ]
GaussianProcessRegressor 의 정답률 :  [-7.52040052 -6.99547982 -4.36356104 -6.2558745  -5.57336905]
GeneralizedLinearRegressor 의 정답률 :  [0.50391368 0.66408094 0.69121537 0.6425692  0.61304926]
GradientBoostingRegressor 의 정답률 :  [0.64527786 0.92096871 0.80441622 0.90488381 0.91341018]
HistGradientBoostingRegressor 의 정답률 :  [0.87805211 0.90751513 0.7321378  0.86098976 0.74435935]
HuberRegressor 의 정답률 :  [0.65634371 0.59149273 0.70451412 0.62466922 0.60917395]
IsotonicRegression 의 정답률 :  [nan nan nan nan nan]
KNeighborsRegressor 의 정답률 :  [0.46258842 0.57734606 0.61976732 0.30816277 0.27281565]
KernelRidge 의 정답률 :  [0.66879845 0.70315473 0.41873534 0.77033021 0.63158694]
Lars 의 정답률 :  [0.62411403 0.73043467 0.54919906 0.68853009 0.57745618]
LarsCV 의 정답률 :  [0.53476898 0.78181047 0.66737786 0.77883696 0.57910432]
Lasso 의 정답률 :  [0.67461958 0.72988176 0.58687087 0.52807674 0.64840441]
LassoCV 의 정답률 :  [0.69047613 0.63596606 0.70313927 0.64829125 0.60269162]
LassoLars 의 정답률 :  [-0.03027148 -0.0131042  -0.08231699 -0.02380741 -0.00857679]
LassoLarsCV 의 정답률 :  [0.60648376 0.81010706 0.62187783 0.73216573 0.6933681 ]
LassoLarsIC 의 정답률 :  [0.6428047  0.70930654 0.66176715 0.74954958 0.70939697]
LinearRegression 의 정답률 :  [0.61093627 0.74341171 0.62998414 0.73342201 0.70877851]
LinearSVR 의 정답률 :  [-1.52811784  0.35484512  0.62327742  0.38389905  0.55390818]
MLPRegressor 의 정답률 :  [0.45224164 0.3398124  0.62422069 0.66349118 0.73627559]
MultiTaskElasticNet 의 정답률 :  [nan nan nan nan nan]
MultiTaskElasticNetCV 의 정답률 :  [nan nan nan nan nan]
MultiTaskLasso 의 정답률 :  [nan nan nan nan nan]
MultiTaskLassoCV 의 정답률 :  [nan nan nan nan nan]
NuSVR 의 정답률 :  [0.2964431  0.29262939 0.18886489 0.07624849 0.37167766]
OrthogonalMatchingPursuit 의 정답률 :  [0.47306865 0.58104555 0.40023557 0.51131561 0.58677819]
OrthogonalMatchingPursuitCV 의 정답률 :  [0.62746248 0.59860815 0.63196967 0.70036064 0.60356457]
PLSCanonical 의 정답률 :  [-1.63980843 -2.90255881 -2.96423453 -2.26236277 -2.1363206 ]
PLSRegression 의 정답률 :  [0.66694769 0.77521605 0.61692893 0.5946487  0.62258673]
PassiveAggressiveRegressor 의 정답률 :  [-0.44177192 -0.09196229 -0.47694849  0.08295173  0.20795721]
PoissonRegressor 의 정답률 :  [0.76911138 0.57504965 0.64847037 0.79310127 0.80816384]
RANSACRegressor 의 정답률 :  [-0.60668639  0.26237197  0.53205883  0.3891158   0.68412663]
RandomForestRegressor 의 정답률 :  [0.92676623 0.83422687 0.78325713 0.91029382 0.75205193]
Ridge 의 정답률 :  [0.68404509 0.6489355  0.73764124 0.58317736 0.79228187]
RidgeCV 의 정답률 :  [0.79667269 0.49009622 0.64643285 0.7231571  0.73827455]
SGDRegressor 의 정답률 :  [-3.02682458e+26 -4.93339499e+26 -1.34796183e+26 -1.48108905e+27
 -1.10969355e+26]
SVR 의 정답률 :  [0.13991755 0.28065067 0.25515272 0.09718461 0.20917627]
TheilSenRegressor 의 정답률 :  [0.69110748 0.60856506 0.72542864 0.53902275 0.69273451]
TransformedTargetRegressor 의 정답률 :  [0.73456788 0.6269797  0.78795753 0.63803763 0.65208217]
TweedieRegressor 의 정답률 :  [0.60753648 0.68182655 0.6854187  0.61632068 0.49834436]
_SigmoidCalibration 의 정답률 :  [nan nan nan nan nan]
'''