from tensorflow.keras.applications import VGG16, VGG19, Xception, ResNet50, ResNet50V2, ResNet101, ResNet101V2, ResNet152, InceptionV3, InceptionResNetV2, MobileNet, MobileNetV2, DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications import NASNetLarge, NASNetMobile
from tensorflow.keras.layers import Dense, Flatten, Input, BatchNormalization, Dropout, Activation
from tensorflow.keras.models import Sequential, Model

model=NASNetLarge()                     

model.summary()

print('동결하기 전 훈련되는 가중치의 수 : ', len(model.trainable_weights)) 
# print('동결한 후 훈련되는 가중치의 수 : ', len(vgg16.trainable_weights)) # 0

## 모델 별로 가장 순수했을 때의 파라미터의 갯수와 가중치 수를 정리하시오
'''
VGG16
input_shape : (224, 224, 3)
Total params: 138,357,544
동결하기 전 훈련되는 가중치의 수 :  32

VGG19
input_shape : (224, 224, 3)
Total params: 143,667,240
동결하기 전 훈련되는 가중치의 수 :  38

Xception
input_shape : (299, 299, 3)
Total params: 22,910,480
동결하기 전 훈련되는 가중치의 수 :  156

ResNet50
input_shape : (224, 224, 3)
Total params: 25,636,712
동결하기 전 훈련되는 가중치의 수 :  214

ResNet101
input_shape : (224, 224, 3)
Total params: 44,707,176
동결하기 전 훈련되는 가중치의 수 :  418

InceptionV3
input_shape : (299, 299, 3)
Total params: 23,851,784
동결하기 전 훈련되는 가중치의 수 :  190

InceptionResNetV2
input_shape : (299, 299, 3)
Total params: 55,873,736
동결하기 전 훈련되는 가중치의 수 :  490

MobileNet
input_shape : (224, 224, 3)
Total params: 4,253,864
동결하기 전 훈련되는 가중치의 수 :  83

MobileNetV2
input_shape : (224, 224, 3)
Total params: 3,538,984
동결하기 전 훈련되는 가중치의 수 :  158

DenseNet121
input_shape : (224, 224, 3)
Total params: 8,062,504
동결하기 전 훈련되는 가중치의 수 :  364

NASNetLarge
input_shape : 
Total params: 88,949,818
동결하기 전 훈련되는 가중치의 수 :  1018

NASNetMobile
input_shape : 
Total params: 5,326,716
동결하기 전 훈련되는 가중치의 수 :  742

'''

