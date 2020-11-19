import numpy as np
import pandas as pd


datasets=pd.read_csv('./data/csv/iris_ys.csv', header=0, index_col=0, sep=',')
#header=0 -> 헤더가 어디에 있는지 명시. 0이면 위에서 첫번째, None으로 하면 헤더도 데이터로 취급.
#index_col -> 인덱스가 어디에 있는지 명시. 0이면 왼쪽에서 첫번째. None이면 없기때문에 인덱스를 따로 만듦
#sep -> 데이터가 뭘로 구분되어 있는가.

print(datasets)
print(datasets.shape) #(150, 5)

#index_col=None, 0, 1 / header = None, 0, 1일 때 shape
'''
                index_col=None    |   index_col=0   |   index_col=1
header=None        (151, 6)       |     (151, 5)    |    (151, 5)
header=0           (150, 6)       |     (150, 5)    |    (150, 5)
header=1           (149, 6)       |     (149, 5)    |    (149, 5)
'''

print(datasets.head()) #위에서 다섯개 데이터
print(datasets.tail()) #끝에서 다섯개 데이터

print(type(datasets)) #<class 'pandas.core.frame.DataFrame'>

#datasets를 numpy로 바꿀 것
aaa=datasets.to_numpy() 
#aaa=datasets.values()

print(type(aaa)) #<class 'numpy.ndarray'>
print(aaa.shape) #(150, 5)

np.save('./data/iris_ys_pd.npy', arr=aaa)