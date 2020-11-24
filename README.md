# bit_seoul
인공지능 개발자 과정

<hr />

## 딥러닝 : 케라스 기초
### 1. 모델링의 순서
1. 데이터 (정제된 데이터: x, y)     
2. 모델 구성    
3. 컴파일, 훈련    
4. 평가, 예측    
[소스코드보기 : 모델 구성](https://github.com/maiorem/bit_seoul/blob/main/Study/keras/keras01.py)     
=> DeepLearning은 정제된 x, y 데이터로 최적의 weight(가중치)와 최소의 loss(손실)을 구하여 예측값을 찾는 것이다.     
=> 기본적인 수식 : y=wx+b      

* 하이퍼 파라미터 튜닝이란?
 최적의 weight와 최소의 loss를 찾기 위해 모델 구조를 튜닝하는 것.           
 현재 모델의 튜닝 :       
 -노드와 레이어 변형        
 -epochs(훈련시키는 횟수)        
 -batch_size 알맞게 쪼개기       
 -검증 데이터 활용하기        
 
* 무조건 노드와 레이어가 많은 것이 좋은가?        
[반례](https://github.com/maiorem/bit_seoul/blob/main/Study/keras/keras08_r2_bad.py)        

### 2. 모델 SAVE와 LOAD, 시각화
 * [적절한 값이 나오는 모델 저장하기](https://github.com/maiorem/bit_seoul/blob/main/Study/keras/keras30_save.py)            
 * [저장한 모델 불러와서 커스텀하기](https://github.com/maiorem/bit_seoul/blob/main/Study/keras/keras31_load.py)         
 * [모델링 시각화-Matplotlib](https://github.com/maiorem/bit_seoul/blob/main/Study/keras/keras32_hist2_graph.py)                
 * [모델링 시각화-Tensorboard](https://github.com/maiorem/bit_seoul/blob/main/Study/keras/keras33_tensorboard.py)            
 * [모델과 가중치를 함께 저장하기 : fit 이후에 세이브](https://github.com/maiorem/bit_seoul/blob/main/Study/keras/keras50_1_save_model.py)         
 * [가중치만 저장하기 : save_weights](https://github.com/maiorem/bit_seoul/blob/main/Study/keras/keras51_1_save_weights.py)        
 * [**모델 체크포인트** : 이전보다 최적의 값이 나올 때마다 저장](https://github.com/maiorem/bit_seoul/blob/main/Study/keras/keras48_ModelCheckPoint.py)         

### 3. 선형회귀(Linear Regression) 
![linear](https://upload.wikimedia.org/wikipedia/commons/thumb/b/be/Normdist_regression.png/300px-Normdist_regression.png)     
(그림출처 : 위키백과)     

* 용어와 함수 정리       
loss(=cost) : 손실함수. 데이터와 예측치의 오차. 주로 mse 혹은 mae를 사용한다.     
optimizer : 최적화 함수.       
metrics : evaluate 함수에 기본적으로 loss가 반환되는 것 외에 구하고자 하는 값이 있을 때 사용. 리스트 형태로 반환.      
:: MSE(Mean Squared Error) : 평균제곱오차. 손실에 제곱을 해서 오차값의 양수와 음수 상쇄를 막는다.      
:: MAE(Mean Absolute Error) : 평균절대오차. 손실의 절대값을 구하여 오차값의 양수와 음수 상쇄를 막는다.     
:: RMSE : 사용자 정의 함수. 평균제곱오차의 제곱근을 구한다.(사이킷런에서 임포트)     
:: R2 : 예측의 적합도를 0과 1 사이의 값으로 표현함. 선형회귀에서 accuracy 대신 많이 사용한다.     
[예측하기](https://github.com/maiorem/bit_seoul/blob/main/Study/keras/keras02_predict.py)       
[metrics의 활용](https://github.com/maiorem/bit_seoul/blob/main/Study/keras/keras04_metrics.py)        
[metrics의 활용2](https://github.com/maiorem/bit_seoul/blob/main/Study/keras/keras04_metrics2.py)         
[RMSE](https://github.com/maiorem/bit_seoul/blob/main/Study/keras/keras06_RMSE.py)        
[R2](https://github.com/maiorem/bit_seoul/blob/main/Study/keras/keras07_r2.py)         

* 훈련 데이터와 테스트 데이터의 분리        
-x의 값이 y가 나오도록 훈련시키는 데이터와 이를 테스트하는 평가 데이터는 분리하는 것이 좋다.        
[train_test](https://github.com/maiorem/bit_seoul/blob/main/Study/keras/keras05_train_test.py)       
[train_test2](https://github.com/maiorem/bit_seoul/blob/main/Study/keras/keras05_train_test2.py)         
-훈련을 검증된 데이터로 하도록 데이터를 또 따로 분리하여 설정할 수 있다.      
validation_data => 검증 할 데이터를 직접 넣어 줌        
validation_split => 훈련 데이터에서 검증 데이터를 잘라서 사용     
[validation_data](https://github.com/maiorem/bit_seoul/blob/main/Study/keras/keras09_val.py)       
[validation_split](https://github.com/maiorem/bit_seoul/blob/main/Study/keras/keras09_val2.py)         
-사이킷런의 train_test_split 함수로 훈련 혹은 테스트 데이터 사이즈 비율을 지정해서 잘라 사용할 수 있음
[train_test_split](https://github.com/maiorem/bit_seoul/blob/main/Study/keras/keras11_train_test_split.py)       
[train_test_split(validation 데이터를 또 분리한 경우)](https://github.com/maiorem/bit_seoul/blob/main/Study/keras/keras11_train_test_split2.py) 

* 다중 선형(MLP)     
입력되는 x의 값이 여러개의 열로 된 다차원 데이터인 경우(input_dim이 1 이상)       
[다중 입력 다중 출력](https://github.com/maiorem/bit_seoul/blob/main/Study/keras/keras12_mlp.py)       
[다중 입력 1 출력](https://github.com/maiorem/bit_seoul/blob/main/Study/keras/keras12_mlp2.py)       
[1 입력 다중 출력](https://github.com/maiorem/bit_seoul/blob/main/Study/keras/keras12_mlp3.py)          
 => input_shape=(3,) 행무시, 열우선          
 만약에 자료구조가 100x10x3인 3차원 데이터를 입력한다면 input_shape=(10,3)          
 => verbose : 값에 따라 훈련이 출력되는 양식이 다르다.               

* 함수형 모델       
-활성화 함수(activation) :           
 => param의 값 = input(입력 차원의 수) * output(받는 노드의 수) + 노드 수(Bias의 연산값) (y=w(weight)x+b(bias))        
 Sequential 모델이 모델 선언부터 하고 레이어를 추가하는 형태인 것과 달리,
 Model로 선언한 함수형 모델은 인풋과 히든레이어, 아웃풋 레이어를 먼저 쌓고 모델을 선언하여 처음과 끝을 정해줌.        
 [함수형 모델](https://github.com/maiorem/bit_seoul/blob/main/Study/keras/keras15_function.py)         
 => Sequential 모델과 함수형 모델의 summary를 비교하면 같은 자료 형태를 가지고 있음을 알 수 있다.
 
* 앙상블 모델 : 두개 이상의 모델을 각각의 독립성을 지켜주면서 조화된 출력값이 나오도록 하는 모델 구조       
=> [앙상블 기본형-다중 입력, 다중 출력(출력과 입력의 양이 같음)](https://github.com/maiorem/bit_seoul/blob/main/Study/keras/keras16_ensemble.py)            
:: Concatenate, concatenate의 차이(사용법이 다르니 구분할 것!)                
 concatenate([병합할 모델1, 병합할 모델2])           
 Concatenate(axis=1)([병합할 모델1, 병합할 모델2])            
=> [다중 입력, 다중 출력(출력이 입력보다 많은 경우)](https://github.com/maiorem/bit_seoul/blob/main/Study/keras/keras16_ensemble2.py)         
=> **[다중 입력, 단일 출력](https://github.com/maiorem/bit_seoul/blob/main/Study/keras/keras16_ensemble3.py)**            
=> [단일 입력, 다중 출력](https://github.com/maiorem/bit_seoul/blob/main/Study/keras/keras16_ensemble4.py)         

* RNN(Recurrent Neural Network 순환신경망) 모델 : 순차적인 데이터 처리           
=> **LSTM(Long-Short-Term Memory)**            
 RNN에서 가장 성능이 좋음.        
 하나의 데이터에서 다음 데이터를 받을 때마다 연산을 하기 때문에 입력을 어떻게 자를 것인지 설정해줘야 한다.
 예를 들어 [2,3,4]가 있고 이를 하나씩 잘라준다고 하면 2에서 3으로 넘어갈 때 +1, 3에서 4로 갈때 +1, 그렇게 y=[5]가 나오도록 함. 때문에 x=[2,3,4]를 쪼개서 [[2],[3],[4]]의 형태로 변환을 해줘야 한다.       
 형변환 함수 : reshape           
 [LSTM 기본형](https://github.com/maiorem/bit_seoul/blob/main/Study/keras/keras17_LSTM.py)         
 [LSTM의 input_shape를 length와 dim으로 바꾸기](https://github.com/maiorem/bit_seoul/blob/main/Study/keras/keras17_LSTM2.py)            
 [좀 더 다양한 수식을 하는 자료로 LSTM 구성](https://github.com/maiorem/bit_seoul/blob/main/Study/keras/keras17_LSTM3_scale.py)          
-> LSTM외에도 RNN에는 Simple RNN과 GRU 등이 있다. 각각을 이용해서 성능 비교하기.           
  [SimpleRNN : 모델 분석](https://github.com/maiorem/bit_seoul/blob/main/Study/keras/keras18_simpleRNN.py)             
 [SimpleRNN : LSTM과 비교](https://github.com/maiorem/bit_seoul/blob/main/Study/keras/keras18_simpleRNN2_scale.py)         
 [GRU : 모델 분석](https://github.com/maiorem/bit_seoul/blob/main/Study/keras/keras19_GRU.py)          
 [GRU : LSTM과 SimpleRnn과 성능 비교](https://github.com/maiorem/bit_seoul/blob/main/Study/keras/keras19_GRU2_scale.py)          
 -> 함수형 모델로 LSTM 구성        
 [함수형 모델 LSTM](https://github.com/maiorem/bit_seoul/blob/main/Study/keras/keras20_LSTM_hamsu.py)           
 => **조기종료 함수 : Early Stopping**                
 : 훈련을 시키는 도중, 가장 낮은 loss 값이 나왔을 때 훈련을 멈추고 결과값을 출력하도록 하는 함수. patience 값을 조절하여 낮은 loss보다 더 낮은 loss를 찾기 위해 현재의 저장값을 넘길 수 있다.         
 [Early Stopping](https://github.com/maiorem/bit_seoul/blob/main/Study/keras/keras21_LSTM_earlyStopping.py)             
 => ?? : LSTM을 레이어에 두개씩 쓰면 성능이 더 좋아지지 않을까?           
 [LSTM을 연속해서 썼을 때의 문제점과 해결 방법](https://github.com/maiorem/bit_seoul/blob/main/Study/keras/keras22_Return_sequence.py)            
 : # LSTM은 2차원을 3차원으로 형변환하여 입력하였으므로 3차원이지만 return_sequences가 디폴트로 False로 주어져 있기에 마지막 출력 시퀀스에 맞춘 차원을 반환한다.               
때문에 LSTM이 한번 더 나오면 요구되는 입력값이 3차원인데 2차원만 입력되는 경우가 발생함.          
그럴 땐 앞에 쓴 LSTM 함수의 return_sequence를 True로 바꿔 본연의 차원인 3차원을 되찾고, 그대로 입력하도록 만든다.              

* CNN모델
  * CNN(Convolution Neural Network) 모델의 파라미터           
 #filter : output. 다음 layer에 10개를 던져준다.                    
 #kernel_size : 이미지를 잘라 연산할 크기 (작업크기)                 
 #strides (디폴트=1) : convolution layer 수행시 한번에 이동할 칸                
 #padding 디폴트는 valid:적용하지 않음 / same : 가장자리에 padding을 씌워 데이터 손실을 막음 (입력 shape가 그대로 들어감. 픽셀은 보통 0으로 채운다)               
 #input_shape=(rows, cols, channels)            
 #입력 형식 : batch_size, rows, cols, channels              
 [CNN 모델의 기본 형식](https://github.com/maiorem/bit_seoul/blob/main/Study/keras/keras35_cnn1.py)        
 Conv2D(filter, (kernel_size가로, kernel_size세로), input_shape=(원본 가로, 원본 세로, 채널)        
 CNN 모델은 입력되는 차원과 출력되는 차원이 동일하므로 LSTM처럼 곧바로 DENSE와 연결시켜 줄 수 없다.      
 사용 기법 : **MaxPooling2D** => 이미지를 중복없이 잘라서 가장 큰 feature를 추출함 / **Flatten** => 다차원으로 구성된 데이터를 일렬로 쭉 펼쳐 1차원으로 만듦          

   
### 4. 다중 분류(Multiple Classification)         
* One Hot Encoding       
   => 데이터를 숫자로 줬을 때, feature의 중요도와 별개로 숫자가 더 큰 값을 컴퓨터가 무조건 더 크다고 판단하는 수가 있다. 그럴 때, True면 1, False면 0으로 값을 줘서 사용자가 원하는 방식으로 컴퓨터가 데이터를 읽도록 출력값을 인코딩 하는 기법 중 하나를 One Hot Encoding이라고 함. 인코딩함으로써 인덱스가 1에 해당하는 분류값을 반환한다. 당연히 y값을 인코딩한다.        
   다중 분류의 몇가지 규칙      
     1. 다중 분류의 output layer의 활성화함수는 **softmax**를 쓴다.      
     2. 다중분류에선 반드시 loss를 **categorical_crossentropy**로 쓴다.           
   
   softmax :  입력받은 값을 0~1사이의 값으로 모두 정규화하며 출력 값들의 총합은 항상 1이 되는 활성화 함수이다.        
   categorical_crossentropy : 다중분류손실함수. 출력값이 one-hot-encoding 된 결과로 나오므로 y값은 무조건 one-hot encoding해서 넣어줘야 함.          
   [CNN 다중 분류 모델](https://github.com/maiorem/bit_seoul/blob/main/Study/keras/keras36_mnist2_cnn.py)         
  
### 5. 이진 분류(Binary Classification)      
 * 다중분류가 여러가지 카테고리를 분류하는 것이라면 이진분류는 0이냐 1이냐, 둘 중 하나를 고르는 것이다.       
 * 활성화 함수 : **sigmoid**         
 * loss 함수는 **binary_crossentropy**      
 예시 : [breast-cancer 데이터셋](https://github.com/maiorem/bit_seoul/blob/main/Study/keras/keras46_cancer_1_dnn.py)           
 
 
## 머신러닝 : 사이킷 기초
### XOR 게이트 : 인공지능의 두번째 겨울
=> 머신러닝으로 구현되는 인공지능이 AND와 OR의 데이터는 학습할 수 있지만, 같은 수는 0, 다른 수는 1로 출력하는 XOR의 문제의 경우 제대로 풀어내지 못하는 문제가 발생.        
(현대의 딥러닝의 경우, 레이어를 쌓아 연산을 늘리면 빠르게 해결이 가능하다)          
[머신러닝의 기본 모델(분류)](https://github.com/maiorem/bit_seoul/blob/main/Study/ml/m05_iris.py)            
[머신러닝의 기본 모델(회귀)](https://github.com/maiorem/bit_seoul/blob/main/Study/ml/m07_boston.py)             
[머신러닝의 분류모델들](https://github.com/maiorem/bit_seoul/blob/main/Study/ml/m11_selectModel.py)          
[머신러닝의 회귀모델들](https://github.com/maiorem/bit_seoul/blob/main/Study/ml/m11_selectModel2.py)              

### Cross-Validation (CV)          
 * 기존 validation data의 문제 : 해당 validation data가 모든 데이터를 대표할 수 있을까?        
 => Cross-validation(교차검증) : train 데이터를 원하는 만큼 split하여 training과 교차로 검증하여 모든 train 데이터를 검사할 수 있도록 함.         

* KFold() : 분할 파라미터를 설정하는 함수.           
* cross_val_scores() : 파라미터로 model, feature, target, cv를 받아 교차검증을 수행하는 함수.         
[교차검증 예시](https://github.com/maiorem/bit_seoul/blob/main/Study/ml/m12_kfold.py)            

### GridSearch AND RandomSearch


### Pipeline


### Feature-Importances

 
  
 
 
         
