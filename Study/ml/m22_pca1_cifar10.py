#pca로 축소해서 모델을 완성하시오
#1. 0.95 이상
#2. 1 이상
#mnist dnn과 loss / acc 비교

import numpy as np
from sklearn.decomposition import PCA
from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x=np.append(x_train, x_test, axis=0)
print(x.shape) #(60000, 32, 32, 3)
x=x.reshape(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3]).astype('float32')/255.

# #PCA로 컬럼 걸러내기
# pca=PCA()
# pca.fit(x)
# cumsum=np.cumsum(pca.explained_variance_ratio_) #누적된 합 표시
# # print(cumsum)

# d=np.argmax(cumsum >= 1) + 1
# # print(cumsum>=0.95) 
# print(d) # 217 # 3072

pca1=PCA(n_components=0.95)
x=pca1.fit_transform(x)
# print(x.shape) #(60000, 217)

x_train=x[:50000, :]
x_test=x[50000:, :]

# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)


y_train=to_categorical(y_train)
y_test=to_categorical(y_test)


# x_train=x_train.astype('float32')/255.
# x_test=x_test.astype('float32')/255.



#2. 모델
model=Sequential()
model.add(Dense(2000, activation='relu', input_shape=(x.shape[1],)))
model.add(Dense(4000, activation='relu'))
model.add(Dense(3000, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='softmax')) 

model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

es=EarlyStopping(monitor='val_loss', patience=50, mode='auto')
model.fit(x_train, y_train, epochs=10000, batch_size=1000, verbose=1, validation_split=0.2, callbacks=[es])

#4. 평가, 예측
loss, accuracy=model.evaluate(x_test, y_test, batch_size=1000)

print('loss : ', loss)
print('accuracy : ', accuracy)


'''
PCA 없는 DNN
loss : 4.863764762878418
acc : 0.40130001306533813
실제값 : [3 8 8 0 6 6 1 6 3 1]
예측값 : [6 9 8 0 4 6 5 2 3 9]

PCA 0.95
loss :  3.6144258975982666
accuracy :  0.5512999892234802

PCA 1
loss :  3.9221692085266113
accuracy :  0.5145000219345093
'''
