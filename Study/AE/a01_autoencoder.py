# 준지도학습
# AutoEncoder : 차원축소, 특성추출
# y 없이 지도하는 비지도학습(pca)과 유사. y는 자기 자신으로 둔다.

import numpy as np
from tensorflow.keras.datasets import mnist

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
(x_train, _), (x_test, _) = mnist.load_data()

x_train=x_train.reshape(60000, 784).astype('float32')/255.
x_test=x_test.reshape(10000, 784)/255.

print(x_train[0])
print(x_test[0])

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

input_img=Input(shape=(784,))
encoded=Dense(64, activation='relu')(input_img) # PCA와 같은 효과. 차원 축소.
decoded=Dense(784, activation='sigmoid')(encoded) # 축소한 차원을 다시 증폭. 0과 1 사이로 전처리

autoencoder = Model(input_img, decoded)
autoencoder.summary()

# autoencoder.compile(optimizer='adam', loss='mse') # sigmoid는 분류가 아닌 전처리를 위해 사용하였으므로 mse도 가능
autoencoder.compile(optimizer='adam', loss='binary_crossentropy') # 이것도 가능

autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, validation_split=0.2) # y는 자기 자신

decoded_img=autoencoder.predict(x_test)


import matplotlib.pyplot as plt

n=10
plt.figure(figsize=(20, 4))
for i in range(n) :
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_img[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
