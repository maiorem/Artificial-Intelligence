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

model=XGBClassifier(max_depth=4)
model.fit(x_train, y_train)

acc=model.score(x_test, y_test)

print("acc : ", acc)

import pickle
pickle.dump(model, open("./save/xgb_save/cancer.pickle.dat", "wb"))
print("저장 완료.")

model2=pickle.load(open("./save/xgb_save/cancer.pickle.dat", "rb"))

print("로드 완료")
acc2=model2.score(x_test, y_test)
print("acc : ", acc2)