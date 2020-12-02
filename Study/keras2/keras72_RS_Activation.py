from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, LSTM, Input, Dropout
from tensorflow.keras.layers import MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.layers import LeakyReLU, ReLU, Softmax, Activation
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad, RMSprop, SGD, Nadam
from tensorflow.keras.activations import relu, elu, selu
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_predict=x_test[:10, :, :]

#1. 데이터 전처리
y_train=to_categorical(y_train) 
y_test=to_categorical(y_test)

x_train=x_train.reshape(60000,28*28).astype('float32')/255.
x_test=x_test.reshape(10000,28*28).astype('float32')/255.
x_predict=x_predict.reshape(10, 28*28).astype('float32')/255.


#2. 모델
def build_model(drop=0.5, optimizer=Adam, lr=0.001, node_num=128, activation='relu') :
    inputs=Input(shape=(28*28,), name='input')
    x=Dense(node_num, name='hidden1')(inputs)
    x=Activation(activation=activation)(x)
    x=Dropout(drop)(x)
    x=Dense(node_num, name='hidden2')(x)
    x=Activation(activation=activation)(x)
    x=Dropout(drop)(x)
    x=Dense(node_num, name='hidden3')(x)
    x=LeakyReLU(alpha=0.3)(x)
    x=Dropout(drop)(x)
    outputs=Dense(10, activation='softmax', name='outputs')(x)
    model=Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer(lr=lr), metrics=['acc'], loss='categorical_crossentropy')
    return model


def create_hyperparameter() :
    batches=[10, 20, 30, 40, 50]
    epoch=[20, 50, 100]
    optimizers=[Adam, RMSprop, Adadelta]
    dropout=[0.1, 0.2, 0.3, 0.4, 0.5]
    lr=[0.001, 0.0001, 0.1]
    node_num=[64, 128, 256, 512]
    activation=['relu','elu','selu']
    return {"batch_size" : batches, "optimizer" : optimizers, "drop" : dropout, "lr": lr, "node_num":node_num, "activation" : activation}

hyperparameters=create_hyperparameter()

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor #그리드서치는 사이킷런 모델만 받을 수 있으므로 케라스모델을 사이킷런으로 랩핑하는 클래스 소환
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

model=KerasClassifier(build_fn=build_model) #케라스 모델을 사이킷런 모델로 랩핑

search=RandomizedSearchCV(model, hyperparameters, cv=3)
search.fit(x_train, y_train)

print(search.best_params_)
acc=search.score(x_test, y_test)
print("최종 스코어 :", acc)

'''
{'optimizer': <class 'tensorflow.python.keras.optimizer_v2.adam.Adam'>, 'node_num': 128, 'lr': 0.001, 'drop': 0.2, 'batch_size': 10, 'activation': 'elu'}
최종 스코어 : 0.9502999782562256
'''