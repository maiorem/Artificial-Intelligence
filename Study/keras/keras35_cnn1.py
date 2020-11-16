#CNN 모델

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten # 레이어를 쫙 펴 줌(1차원으로 변환)
#Conv 파라미터 :
#filter : output. 다음 layer에 10개를 던져준다.
#kernel_size : 이미지를 잘라 연산할 크기 (작업크기)
#strides (디폴트=1) : convolution layer 수행시 한번에 이동할 칸
#padding 디폴트는 valid:적용하지 않음 / same : 가장자리에 padding을 씌워 데이터 손실을 막음 (입력 shape가 그대로 들어감. 픽셀은 보통 0으로 채운다)
#input_shape=(rows, cols, channels)
#입력 형식 : batch_size, rows, cols, channels

#activation 디폴트 = relu


#참고) LSTM의 파라미터 :
#units : 출력해주는 노드 수
#return_sequence : 디폴트는 False
#입력 형식 : batch_size, timesteps, feature
#input_shape=(timesteps, feature)
#activation 디폴트 = tanh

model=Sequential()
#(2,2) <= 이미지를 잘라 연산할 크기 / (10,10,1) 원본이미지 10x10에 흑백
model.add(Conv2D(10, (2,2), input_shape=(10,10,1))) #흑백이면 1, 칼라면 3
                    #kernel_size, strides
#conv는 lstm과 달리 엮을 수록 이미지가 증폭되므로 성능이 좋아짐
model.add(Conv2D(5, (2,2), padding='same')) #shape=(9,9,10) => 10x10 이미지를 두개씩 잘랐으므로 9x9가 됨. 앞의 출력값이 입력되어 9x9 이미지가 총 10개. 여기서는 padding을 same으로 줬으므로 9x9x5를 댜음 레이어로 출력
model.add(Conv2D(3, (3,3), padding='valid')) #다음 레이어에 넘겨주는 출력 7x7x3
model.add(Conv2D(7, (2,2))) #다음 레이어로 6x6x7 전달
# 10(필터 수) - 2(커널 사이즈) + 1(stride) = 9 /

# lstm과 달리, Conv가 출력하는 차원은 입력과 동일하다. Dense와 그냥 엮어줄 수 없음.

#CNN 기법
#MaxPooling2D : 이미지를 중복없이 잘라서 가장 큰 feature를 추출
model.add(MaxPooling2D()) #현재 shape : 3x3x7 (중복 없이 2씩 잘라서)
model.add(Flatten())    #3*3*7=63 / (63,)로 변환
model.add(Dense(1)) #최종 아웃풋

model.summary()
# Conv Parameter : 
# (필터 수 x 커널사이즈(가로) x 커널사이즈(세로) + bias) x 다음 레이어로 전달하는 output 노드의 갯수