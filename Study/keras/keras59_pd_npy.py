import pandas as pd
import numpy as np

# iris_y2.csv 파일을 넘파이로 불러오기
# 불러온 데이터를 판다스로 저장하시오
datasets=np.loadtxt('./data/csv/iris_ys2.csv', delimiter=",")
print(type(datasets))
print(datasets.shape)

x=datasets[:,:4]
y=datasets[:,4]

df=pd.DataFrame(data=datasets, index=None, columns=None)
print(type(df))
print(df.shape)

df.to_csv('./data/csv/iris_ys2_pd.csv', mode='a', header=False)

print(x.shape)
print(y.shape)