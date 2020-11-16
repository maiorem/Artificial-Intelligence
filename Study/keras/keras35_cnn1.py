from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D

model=Sequential()
#(2,2) <= 이미지를 잘라 연산할 크기 / (5,5,1) 원본이미지 5x5에 흑백
model.add(Conv2D(10, (2,2), input_shape=(5,5,1))) #흑백이면 1, 칼라면 3
                    #kernel_size, strides
#filter : output. 다음 layer에 10개를 던져준다.
#kernel_size : 이미지를 잘라 연산할 크기 (작업크기)
#strides (디폴트=1) : convolution layer 수행시 한번에 이동할 칸
#padding 디폴트는 valid:적용하지 않음 / same : 가장자리에 padding을 씌워 데이터 손실을 막음 (입력 shape가 그대로 들어감. 픽셀은 보통 0으로 채운다)
#input_shape=(rows, cols, channels)
#입력 형식 : batch_size, rows, cols, channels

#참고) LSTM의 파라미터 :
#units : 출력해주는 노드 수
#return_sequence : 디폴트는 False
#입력 형식 : batch_size, timesteps, feature
#input_shape=(timesteps, feature)
