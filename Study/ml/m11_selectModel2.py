# 리그레서 모델들 추출

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils.testing import all_estimators
import warnings

warnings.filterwarnings('ignore')

iris=pd.read_csv('./data/csv/boston_house_prices.csv', header=1, index_col=None)

x=iris.iloc[:, 0:-1]
y=iris.iloc[:, -1]

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=66)

allAlgorithms=all_estimators(type_filter='regressor')

for (name, algorithm) in allAlgorithms :
    try :
        model=algorithm()
        model.fit(x_train, y_train)
        y_pred=model.predict(x_test)
        print(name, '의 정답률 : ', r2_score(y_test, y_pred))
    except :
        pass

import sklearn
print(sklearn.__version__) #0.22.1 버전에 문제 있어서 출력이 안됨.

'''
ARDRegression 의 정답률 :  0.8012569266997763
AdaBoostRegressor 의 정답률 :  0.9076727410627611
BaggingRegressor 의 정답률 :  0.907908378420125
BayesianRidge 의 정답률 :  0.7937918622384752
CCA 의 정답률 :  0.7913477184424631
DecisionTreeRegressor 의 정답률 :  0.8193000646460461
DummyRegressor 의 정답률 :  -0.0005370164400797517
ElasticNet 의 정답률 :  0.7338335519267194
ElasticNetCV 의 정답률 :  0.7167760356856181
ExtraTreeRegressor 의 정답률 :  0.6545640780127913
ExtraTreesRegressor 의 정답률 :  0.9307400679657453
GammaRegressor 의 정답률 :  -0.0005370164400797517
GaussianProcessRegressor 의 정답률 :  -6.073105259620457
GeneralizedLinearRegressor 의 정답률 :  0.7442833362029138
**GradientBoostingRegressor 의 정답률 :  0.9452385394613945
HistGradientBoostingRegressor 의 정답률 :  0.9323597806119726
HuberRegressor 의 정답률 :  0.7551817913064872
KNeighborsRegressor 의 정답률 :  0.5900872726222293
KernelRidge 의 정답률 :  0.8333325494049566
*Lars 의 정답률 :  0.7746736096721594
LarsCV 의 정답률 :  0.7981576314184007
Lasso 의 정답률 :  0.7240751024070102
LassoCV 의 정답률 :  0.7517507753137198
LassoLars 의 정답률 :  -0.0005370164400797517
LassoLarsCV 의 정답률 :  0.8127604328474287
LassoLarsIC 의 정답률 :  0.8131423868817642
LinearRegression 의 정답률 :  0.8111288663608649
LinearSVR 의 정답률 :  0.7756737078973176
MLPRegressor 의 정답률 :  0.43545375241983386
NuSVR 의 정답률 :  0.2594558622083819
OrthogonalMatchingPursuit 의 정답률 :  0.5827617571381449
OrthogonalMatchingPursuitCV 의 정답률 :  0.78617447738729
PLSCanonical 의 정답률 :  -2.2317079741425765
PLSRegression 의 정답률 :  0.8027313142007888
PassiveAggressiveRegressor 의 정답률 :  0.0630527153564201
PoissonRegressor 의 정답률 :  0.8575675972952628
RANSACRegressor 의 정답률 :  0.37695312632576705
RandomForestRegressor 의 정답률 :  0.9196612724116489
*Ridge 의 정답률 :  0.8098487632912243
RidgeCV 의 정답률 :  0.8112529182634152
SGDRegressor 의 정답률 :  -2.2582275146157208e+26
SVR 의 정답률 :  0.2347467755572229
TheilSenRegressor 의 정답률 :  0.7938635456584675
TransformedTargetRegressor 의 정답률 :  0.8111288663608649
TweedieRegressor 의 정답률 :  0.7442833362029138
'''
