from numpy import array
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Input
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping


x=array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
        [5,6,7],[6,7,8],[7,8,9],
        [8,9,10],[9,10,11], [10,11,12],
        [2000,3000,4000],[3000,4000,5000],[4000,5000,6000],
        [100,200,300]])

y=array([4,5,6,7,8,9,10,11,12,13,5000,6000,7000,400])

x_predict=array([55,65,75])
x_predict2=array([6600,6700,6800]) #max 이상인 데이터의 처리


x_predict=x_predict.reshape(1,3)
x_predict2=x_predict2.reshape(1,3)


#데이터 전처리 3. Robust Scaler
from sklearn.preprocessing import RobustScaler

scaler=RobustScaler()
scaler.fit(x) #fit은 train만 한다 (이미 scaler에 저장된 fit을 사용)
x=scaler.transform(x)
x_predict=scaler.transform(x_predict)
x_predict2=scaler.transform(x_predict2)


print(x)
print(x_predict)
print(x_predict2)
