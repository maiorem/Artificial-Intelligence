import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist 

(x_train, y_train), (x_test, y_test)=mnist.load_data()

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)
print(x_train[500])
print(y_train[500])

plt.imshow(x_train[500], 'Blues')
plt.show()
