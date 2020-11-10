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



