# bit_seoul
인공지능 개발자 과정

<hr />

## 케라스 기초
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


### 2. 선형회귀(Linear Regression) 
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
=> [앙상블 기본형(다중 입력, 다중 출력(출력과 입력의 양이 같음)](https://github.com/maiorem/bit_seoul/blob/main/Study/keras/keras16_ensemble.py)            
=> [다중 입력, 다중 출력(출력이 입력보다 많은 경우)](https://github.com/maiorem/bit_seoul/blob/main/Study/keras/keras16_ensemble2.py)         
=> **[다중 입력, 단일 출력](https://github.com/maiorem/bit_seoul/blob/main/Study/keras/keras16_ensemble3.py)**            
=> [단일 입력, 다중 출력](https://github.com/maiorem/bit_seoul/blob/main/Study/keras/keras16_ensemble4.py)            



