# 기준 : XGbost
# 1. feature importance 0 제거
# 2. 하위 30% 제거
# 3. 디폴트와 성능 비교
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA

dataset=load_breast_cancer()
x=dataset.data
y=dataset.target

print(x.shape)



x_train, x_test, y_train, y_test=train_test_split(x, y, train_size=0.8)

model=XGBClassifier(max_depth=4)
model.fit(x_train, y_train)

y_predict=model.predict(x_test)

print("acc :", model.score(x_test, y_test))
print(model.feature_importances_)

'''
Default
acc : 0.956140350877193
[0.         0.03477669 0.         0.         0.03392176 0.00429625
 0.02141644 0.05427442 0.00079031 0.00389343 0.         0.00070328
 0.         0.00469953 0.005103   0.00899227 0.         0.0057787
 0.00790907 0.         0.04495754 0.01834395 0.19929771 0.31475484
 0.0097423  0.00501918 0.03719619 0.17019312 0.00120304 0.01273711]

0 제거
acc : 0.956140350877193
[0.34671882 0.04266788 0.02599428 0.1061023  0.08311348 0.01577053
 0.04079856 0.03661584 0.01779494 0.01519207 0.07767107 0.03647927
 0.01609398 0.00322795 0.02066069 0.04047965 0.02586412 0.01620531
 0.00797606 0.00737984 0.00631428 0.00389191 0.00698718]

30% 제거
acc : 0.9649122807017544
[0.3121231  0.03653114 0.03054608 0.10644057 0.14944103 0.01128189
 0.08187144 0.01103715 0.00688865 0.02402719 0.0559271  0.02868302
 0.01293857 0.0325514  0.01440136 0.02878833 0.02447723 0.0159372
 0.00730527 0.0088023 ]
# '''