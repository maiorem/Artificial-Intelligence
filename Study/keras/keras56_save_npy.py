# 나머지 데이터셋을 저장하시오
from sklearn.datasets import load_boston, load_breast_cancer, load_diabetes, load_iris
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import cifar10, cifar100
import numpy as np

iris_dataset=load_iris()
iris_x=iris_dataset.data
iris_y=iris_dataset.target
iris_x_train, iris_x_test, iris_y_train, iris_y_test=train_test_split(iris_x, iris_y, test_size=0.2)

np.save('./data/iris_x_train.npy', arr=iris_x_train)
np.save('./data/iris_x_test.npy', arr=iris_x_test)
np.save('./data/iris_y_train.npy', arr=iris_y_train)
np.save('./data/iris_y_test.npy', arr=iris_y_test)



boston_dataset=load_boston()
boston_x=boston_dataset.data
boston_y=boston_dataset.target
boston_x_train, boston_x_test, boston_y_train, boston_y_test=train_test_split(boston_x, boston_y, test_size=0.2)

np.save('./data/boston_x_train.npy', arr=boston_x_train)
np.save('./data/boston_x_test.npy', arr=boston_x_test)
np.save('./data/boston_y_train.npy', arr=boston_y_train)
np.save('./data/boston_y_test.npy', arr=boston_y_test)


diabetes_dataset=load_diabetes()
diabetes_x=diabetes_dataset.data
diabetes_y=diabetes_dataset.target
diabetes_x_train, diabetes_x_test, diabetes_y_train, diabetes_y_test=train_test_split(diabetes_x, diabetes_y, test_size=0.2)

np.save('./data/diabetes_x_train.npy', arr=diabetes_x_train)
np.save('./data/diabetes_x_test.npy', arr=diabetes_x_test)
np.save('./data/diabetes_y_train.npy', arr=diabetes_y_train)
np.save('./data/diabetes_y_test.npy', arr=diabetes_y_test)

cancer_dataset=load_breast_cancer()
cancer_x=cancer_dataset.data
cancer_y=cancer_dataset.target
cancer_x_train, cancer_x_test, cancer_y_train, cancer_y_test=train_test_split(cancer_x, cancer_y, test_size=0.2)

np.save('./data/cancer_x_train.npy', arr=cancer_x_train)
np.save('./data/cancer_x_test.npy', arr=cancer_x_test)
np.save('./data/cancer_y_train.npy', arr=cancer_y_train)
np.save('./data/cancer_y_test.npy', arr=cancer_y_test)



(cifar10_x_train, cifar10_y_train), (cifar10_x_test, cifar10_y_test) = cifar10.load_data()
(cifar100_x_train, cifar100_y_train), (cifar100_x_test, cifar100_y_test) = cifar100.load_data()

np.save('./data/cifar10_x_train.npy', arr=cifar10_x_train)
np.save('./data/cifar10_x_test.npy', arr=cifar10_x_test)
np.save('./data/cifar10_y_train.npy', arr=cifar10_y_train)
np.save('./data/cifar10_y_test.npy', arr=cifar10_y_test)


np.save('./data/cifar100_x_train.npy', arr=cifar100_x_train)
np.save('./data/cifar100_x_test.npy', arr=cifar100_x_test)
np.save('./data/cifar100_y_train.npy', arr=cifar100_y_train)
np.save('./data/cifar100_y_test.npy', arr=cifar100_y_test)
