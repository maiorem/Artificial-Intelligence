from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Input, BatchNormalization, Dropout, Activation
from tensorflow.keras.models import Sequential, Model

# model=VGG16()                     # 138,357,544
vgg16=VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3)) # 14,714,688
vgg16.trainable=False

vgg16.summary()

print('동결하기 전 훈련되는 가중치의 수 : ', len(vgg16.trainable_weights)) # 32 # 26
# print('동결한 후 훈련되는 가중치의 수 : ', len(vgg16.trainable_weights)) # 0

model=Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(256))
# model.add(BatchNormalization())
# model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Dense(10, activation='softmax'))

model.summary()

print('훈련되는 가중치의 수 : ', len(model.trainable_weights)) # 6
# print(model.trainable_weights)

# model.add(BatchNormalization()) => 8개
# model.add(Dropout(0.2)) => 6개
# model.add(Activation('relu')) => 6개
# ==> Dropout과 Activation은 가중치 연산을 하지 않는다. BatchNormalization은 한다.

import pandas as pd
pd.set_option('max_colwidth', -1)

layers=[(layer, layer.name, layer.trainable) for layer in model.layers]
aaa = pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])

print(aaa)
