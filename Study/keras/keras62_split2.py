import numpy as np 

# 2차원 리스트의 행을 사이즈만큼 잘라 3차원으로 반환하는 split함수
def split_data(x, size) :
    data=[]
    for i in range(x.shape[0]-size+1) :
        data.append(x[i:i+size,:])
    return np.array(data)



# # 테스트 : 2차원 데이터셋
# from sklearn.datasets import load_iris
# dataset=load_iris()
# x=dataset.data #(150,4)
# y=dataset.target #(150,)
# size=5
# # (5,4) 행렬 데이터 146개


# data_iris=split_data(x, size)
# print('======================')
# print(data_iris)
# print(data_iris.shape)