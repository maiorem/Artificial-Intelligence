import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x) :
    return 1 / (1 + np.exp(-x))

x = np.arange(-5, 5, 1)
y = sigmoid(x)

print('x : ', x)
print('y : ', y)

plt.plot(x, y)
plt.grid()
plt.show()

