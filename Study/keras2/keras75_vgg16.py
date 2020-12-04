from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Input
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
model.add(Dense(10, activation='softmax'))

model.summary()

print('동결하기 전 훈련되는 가중치의 수 : ', len(model.trainable_weights)) # 2
