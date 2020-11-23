import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

wine=pd.read_csv('./data/csv/winequality-white.csv', sep=';', header=0)
y=wine['quality']
x=wine.drop('quality', axis=1)

print(x.shape) #(4898, 11)
print(y.shape) #(4898,)

newlist=[]
for i in list(y) :
    if i <=4 :
        newlist+=[0]
    elif i<=7 :
        newlist+=[1]
    else :
        newlist+=[2]

y=newlist

x_train, x_test, y_train, y_test=train_test_split(x, y, train_size=0.8)

scale=StandardScaler()
scale.fit(x_train)
x_train=scale.transform(x_train)
x_test=scale.transform(x_test)



# 2. 모델
# model=LinearSVC()
# model=SVC()
# model=KNeighborsClassifier()
# model=KNeighborsRegressor()
model=RandomForestClassifier()
# model=RandomForestRegressor()


# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
score=model.score(x_test, y_test)

y_predict=model.predict(x_test)
acc=accuracy_score(y_test, y_predict)

print('score :', score)
print('acc :', acc)

print(y_test[:10], '의 예측 결과 ', y_predict[:10])

'''
score : 0.9489795918367347
acc : 0.9489795918367347
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1] 의 예측 결과  [1 1 1 1 1 1 1 1 1 1]
'''
