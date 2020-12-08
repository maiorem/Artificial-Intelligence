from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score, accuracy_score
import time
import pickle

start1=time.time()

x, y=load_boston(return_X_y=True)

x_train, x_test, y_train, y_test=train_test_split(x, y, train_size=0.8)

model=XGBRegressor(n_jobs=-1)
model.fit(x_train, y_train)
score=model.score(x_test, y_test)

print('score :', score)

thresholds=np.sort(model.feature_importances_)
print(thresholds)

for thresh in thresholds :
    selection=SelectFromModel(model, threshold=thresh, prefit=True)
    
    select_x_train=selection.transform(x_train)
    
    selection_model=XGBRegressor(n_jobs=6)
    selection_model.fit(select_x_train, y_train)

    selec_x_test=selection.transform(x_test)
    y_predict=selection_model.predict(selec_x_test)

    score=r2_score(y_test, y_predict)

    print("Thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1], score*100.0))
    pickle.save(open("./save/xbg_save/boston.pickle.dat", "wb"))

end1=time.time() - start1
print("잡스 걸린 시간 : ", end1)

print(model.feature_importances_)


model2=pickle.load(open("./save/xbg_save/boston.pickle.dat", "rb"))
print("로드 완료")
r2=model2.score(x_test, y_test)
print("r2 : ", r2)