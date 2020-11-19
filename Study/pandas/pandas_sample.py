import pandas as pd
import numpy as np
from numpy.random import randn
np.random.seed(100)

data=randn(5,4)
print(data)
df=pd.DataFrame(data, index='A B C D E'.split(), columns='가 나 다 라'.split())
print(df)

data2=[[1,2,3,4], [5,6,7,8],[9,10,11,12], [13,14,15,16], [17,18,19,20]]
df2=pd.DataFrame(data2, index=['A', 'B', 'C', 'D','E'], columns=['가', '나', '다', '라'])

print(df2)

df3=pd.DataFrame(np.array([[1,2,3], [4,5,6]]))
print(df3)

# 컬럼
print("df2['나'] : \n",df2['나']) # 2, 6, 10, 14, 18
print("df2['나', '라'] : \n",df2[['나', '라']]) 
'''
     나   라
A   2   4
B   6   8
C  10  12
D  14  16
E  18  20
'''

# print("df2[0] : ", df2[0]) # 에러 : 컬럼이 존재하지 않음
# print("df2.loc['나'] : \n", df2.loc['나']) # 에러 : loc는 행 우선으로 사용

print("df2.iloc[:, 2] : \n", df2.iloc[:, 2]) # index location # 인덱스 2열의 모든 행
# print("df2[:, 2] : \n", df2[:, 2]) # 에러 : pandas에선 헤더와 열의 위치 기준으로 데이터 사용.


# 로우
print("df2.loc['A'] : \n", df2.loc['A']) # A행의 모든 열
print("df2.loc['A', 'C'] : \n", df2.loc[['A', 'C']]) # A행의 모든 열
'''
    가   나   다   라
A  1   2   3   4
C  9  10  11  12
'''

print("df2.iloc[0] : \n", df2.iloc[0]) # 인덱스 0번째 행
print("df2.iloc[[0, 2]] : \n", df2.iloc[[0, 2]]) 


#행렬
print("df2.loc[['A', 'B'], ['나', '다']] :\n", df2.loc[['A', 'B'], ['나', '다']] )
'''
    나  다
A  2  3
B  6  7
'''

#한개의 값만 확인
print("df2.loc['E', '다'] : \n", df2.loc['E', '다'])    # 19
print("df2.iloc[4, 2] : ", df2.iloc[4,2])               # 19
print("df2.iloc[4][2] : ", df2.iloc[4][2])              # 19

