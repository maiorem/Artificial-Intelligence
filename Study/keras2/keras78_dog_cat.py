from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

img_dog=load_img('./data/dog_cat/dog.jpg', target_size=(224, 224, 3))
img_cat=load_img('./data/dog_cat/cat.jpg', target_size=(224, 224, 3))
img_suit=load_img('./data/dog_cat/suit.jpg', target_size=(224, 224, 3))
img_lion=load_img('./data/dog_cat/lion.jpg', target_size=(224, 224, 3))

# plt.imshow(img_dog)
# plt.show()


arr_dog=img_to_array(img_dog)
arr_cat=img_to_array(img_cat)
arr_suit=img_to_array(img_suit)
arr_lion=img_to_array(img_lion)

print(arr_dog)
print(type(arr_dog)) # <class 'numpy.ndarray'>
print(arr_dog.shape) # (224, 224, 3)

# RGB => BGR
from tensorflow.keras.applications.vgg16 import preprocess_input
arr_dog=preprocess_input(arr_dog)
arr_cat=preprocess_input(arr_cat)
arr_suit=preprocess_input(arr_suit)
arr_lion=preprocess_input(arr_lion)

print(arr_dog.shape) # (224, 224, 3)
print(arr_cat.shape) # (224, 224, 3)

arr_input=np.stack([arr_dog, arr_cat, arr_suit, arr_lion])
print(arr_input.shape) # (2, 224, 224, 3)

#2. 모델 구성
model=VGG16()
probs = model.predict(arr_input)

print(probs)
print('probs.shape : ', probs.shape) # (2, 1000)


# 이미지 결과 확인
from tensorflow.keras.applications.vgg16 import decode_predictions
results=decode_predictions(probs)
print("------------------------------------------")
print('results[0] : ', results[0])
print('results[1] : ', results[1])
print('results[2] : ', results[2])
print('results[3] : ', results[3])
