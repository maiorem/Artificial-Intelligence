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


### 2. 선형회귀(Linear Regression) 모델
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
x의 값이 y가 나오도록 훈련시키는 데이터와 이를 테스트하는 평가 데이터는 분리하는 것이 좋다.        
[train_test](https://github.com/maiorem/bit_seoul/blob/main/Study/keras/keras05_train_test.py)       
[train_test2](https://github.com/maiorem/bit_seoul/blob/main/Study/keras/keras05_train_test2.py)         
훈련을 검증된 데이터로 하도록 데이터를 또 따로 분리하여 설정할 수 있다.      
validation_data => 검증 할 데이터를 직접 넣어 줌        
validation_split => 훈련 데이터에서 검증 데이터를 잘라서 사용     
[validation](https://github.com/maiorem/bit_seoul/blob/main/Study/keras/keras09_val.py)       
[validation2](https://github.com/maiorem/bit_seoul/blob/main/Study/keras/keras09_val2.py)         





