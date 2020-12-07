import numpy as np

def outliers(data_out) :
    quartile_1, quartile_3=np.percentile(data_out, [25, 75]) # 데이터의 25percentile과 75percentile 지점을 각각의 변수에 넣어 준다.
    print("1사분위 :", quartile_1) # 3.25
    print("3사분위 :", quartile_3) # 97.5
    iqr = quartile_3 - quartile_1 # 94.25
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((data_out>upper_bound) | (data_out<lower_bound))

a = np.array([1,2,3,4,10000, 6, 7,5000, 90, 100])

b = outliers(a)
print("이상치의 위치 : ", b) # array([4, 7] # 10000, 5000