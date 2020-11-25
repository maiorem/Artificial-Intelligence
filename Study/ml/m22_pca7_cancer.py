# 이진분류
import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, MaxPooling2D, Dropout, Conv2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

dataset=load_breast_cancer()
x=dataset.data
y=dataset.target
# print(x)
# print(y)
# print(x.shape, y.shape) #(569, 30) (569,)

# #PCA로 컬럼 걸러내기
# pca=PCA()
# pca.fit(x)
# cumsum=np.cumsum(pca.explained_variance_ratio_) #누적된 합 표시
# # print(cumsum)

# d=np.argmax(cumsum >= 1) + 1
# # print(cumsum>=0.95) 
# print(d) # 1 1

pca1=PCA(n_components=0.95)
x=pca1.fit_transform(x)


x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.1)


# scaler=MinMaxScaler()
# scaler.fit(x_train)
# x_train=scaler.transform(x_train)
# x_test=scaler.transform(x_test)

model=Sequential()
model.add(Dense(80, activation='relu', input_shape=(x.shape[1],)))
model.add(Dense(350, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(550, activation='relu'))
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(480, activation='relu'))
model.add(Dense(180, activation='relu'))
model.add(Dense(80))
model.add(Dense(1, activation='sigmoid'))


model.summary()
# model.save("./save/keras46_dnn.h5")
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
es=EarlyStopping(monitor='accuracy', patience=30, mode='auto')
# to_hist=TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True)

model.fit(x_train, y_train, epochs=10000, batch_size=1, validation_split=0.2, callbacks=[es])

#4. 평가, 예측
loss, accuracy=model.evaluate(x_test, y_test, batch_size=1)

print('loss : ', loss)
print('accuracy : ', accuracy)



# y_predict=model.predict(x_test)

# print('실제값 : ', y_test)
# print('예측값 : ', y_predict)

'''
PCA X
loss :  0.043071404099464417
accuracy :  0.9912280440330505

PCA 0.95
loss :  0.237941712141037
accuracy :  0.9385964870452881

PCA 1
loss :  0.3073771595954895
accuracy :  0.9122806787490845
'''
