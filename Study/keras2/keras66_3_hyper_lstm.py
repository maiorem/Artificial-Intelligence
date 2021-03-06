from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, LSTM, Input, Dropout
from tensorflow.keras.layers import MaxPooling2D, Flatten, Activation
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad, RMSprop, SGD, Nadam
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_predict=x_test[:10, :, :]

#1. 데이터 전처리
y_train=to_categorical(y_train) 
y_test=to_categorical(y_test)

x_train=x_train.reshape(60000,28*28, 1).astype('float32')/255.
x_test=x_test.reshape(10000,28*28, 1).astype('float32')/255.
x_predict=x_predict.reshape(10, 28*28, 1).astype('float32')/255.


#2. 모델
## Parameter Tuning
## drop : Dropout
## optimizer : adam, rmsprop, adadelta
## node_num : 64, 128, 256, 512
## activation : 'relu', 'selu', 'elu'
## lr : learning_rate
## epochs
## batch_size
def build_model(drop=0.5, optimizer=Adam, node_num=128, activation='relu', lr=0.001) :
    inputs=Input(shape=(28*28, 1), name='input')
    x=LSTM(node_num, name='hidden1')(inputs)
    x=Activation(activation=activation)(x)
    x=Dropout(drop)(x)
    x=Dense(256, name='hidden2')(x)
    x=Activation(activation=activation)(x)
    x=Dropout(drop)(x)
    x=Dense(128, name='hidden3')(x)
    x=Activation(activation=activation)(x)
    x=Dropout(drop)(x)
    outputs=Dense(10, activation='softmax', name='outputs')(x)
    model=Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer(lr=lr), metrics=['acc'], loss='categorical_crossentropy')
    return model


def create_hyperparameter() :
    batches=[10, 20, 30, 40, 50]
    epoch=[20, 50, 100]
    optimizers=[RMSprop, Adam, Adadelta]
    dropout=[0.1, 0.2, 0.3, 0.4, 0.5]
    lr=[0.001, 0.0001, 0.1]
    node_num=[64, 128, 256, 512]
    return {"batch_size" : batches, "optimizer" : optimizers, "drop" : dropout, "epochs": epoch, "lr": lr, "node_num":node_num}

hyperparameters=create_hyperparameter()

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

model=KerasClassifier(build_fn=build_model)

search=RandomizedSearchCV(model, hyperparameters, cv=3)
search.fit(x_train, y_train)

print(search.best_params_)
acc=search.score(x_test, y_test)
print("최종 스코어 :", acc)

'''
{'optimizer': 'adam', 'drop': 0.1, 'batch_size': 10}
최종 스코어 : 0.9628000259399414

'''