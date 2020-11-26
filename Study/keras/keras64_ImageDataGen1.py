# 남자 여자 구분하기를
# 넘파이 저장
# fit_generator로 코딩

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from sklearn.model_selection import train_test_split

np.random.seed(33)


# 이미지 생성 옵션 정하기
train_datagen = ImageDataGenerator(rescale=1./255, 
                                horizontal_flip=True,
                                vertical_flip=True,
                                width_shift_range=0.1,
                                height_shift_range=0.1,
                                rotation_range=5,
                                zoom_range=1.2,
                                shear_range=0.7,
                                fill_mode='nearest'
                                )
test_datagen = ImageDataGenerator(rescale=1./255)

# flow 또는 flow_from_directory
# 실제 데이터가 있는 곳을 알려주고, 이미지를 불러오는 작업
xy=train_datagen.flow_from_directory(
    './data/data2', #실제 이미지가 있는 폴더는 라벨이 됨. (ad/normal=0/1)
    target_size=(150,150),
    batch_size=200,
    class_mode='binary' 
    #, save_to_dir='./data/data1_2/train' #변환한 파일을 저장
) # x와 y가 이미 갖춰진 데이터셋

np.save('./data/keras64_x.npy', arr=xy[0][0])
np.save('./data/keras64_y.npy', arr=xy[0][1])


# xy_train, xy_test=train_test_split(xy, train_size=0.8)


# # # print(xy_train[0][0].shape) #(10, 300, 300, 3) batch_size=10  # x
# # # print(xy_train[0][1].shape) #(10,)                           # y



# model=Sequential()
# model.add(Conv2D(10, (2,2), input_shape=(150,150,3)))
# model.add(Conv2D(30, (3,3)))
# model.add(Conv2D(20, (2,2)))
# model.add(MaxPooling2D())
# model.add(Flatten())
# model.add(Dense(200, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))


# model.compile(loss='sparse_categorical_crossentopy', optimizer='adam', metrics=['accuracy'])

# # train 셋에 이미 x와 y가 존재하므로 하나만 써주면 됨
# history=model.fit_generator(
#     xy_train,
#     steps_per_epoch=17,
#     epochs=20,
#     validation_data=xy_test, 
#     validation_steps=4
# )

# acc=history.history['accuracy']
# val_acc=history.history['val_accuracy']
# loss=history.history['loss']
# val_loss=history.history['val_loss']

# plt.plot(acc)
# plt.plot(val_acc)
# plt.plot(loss)
# plt.plot(val_loss)

# plt.title('loss & acc')
# plt.ylabel('loss, acc')
# plt.xlabel('epoch')

# plt.legend(['loss', 'val_loss', 'acc', 'val_acc'])
# plt.show()