import numpy as np
import matplotlib.pyplot as plt

def relu(x) :
    return np.maximum(0, x)

x = np.arange(-5, 5, 0.1)
y = relu(x)

plt.plot(x, y)
plt.grid()
plt.show()

# relu 친구들 찾기
# elu : ReLU와 거의 비슷한 형태를 갖습니다. 지수 함수를 이용하여 입력이 0 이하일 경우 부드럽게 깎아줍니다. 미분 함수가 끊어지지 않고 이어져있는 형태를 보입니다. 별도의 알파 값을 파라미터로 받는데 일반적으로 1로 설정됩니다.
# selu 
# LeakyReLU : ReLU와 거의 비슷한 형태를 갖습니다. 입력 값이 음수일 때 완만한 선형 함수를 그려줍니다. 일반적으로 알파를 0.01로 설정합니다. 
# PReLU :  LeakyReLU와 거의 유사한 형태를 보입니다. 하지만 LeakyReLU에서는 알파 값이 고정된 상수였던 반면에 PReLU에서는 학습이 가능한 파라미터로 설정됩니다. 
# ThresholdReLU, 
# GELU
# https://mlfromscratch.com/activation-functions-explained/#/
# https://yeomko.tistory.com/39