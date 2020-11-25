import numpy as np
from sklearn.decomposition import PCA
from tensorflow.keras.datasets import mnist
(x_train, _), (x_test, _) = mnist.load_data()

x=np.append(x_train, x_test, axis=0)
print(x.shape) #(70000, 28, 28)
x=x.reshape(x.shape[0], x.shape[1]*x.shape[2])

pca=PCA()
pca.fit(x)
cumsum=np.cumsum(pca.explained_variance_ratio_) #누적된 합 표시
print(cumsum)

d=np.argmax(cumsum >= 1) + 1
# print(cumsum>=0.95) 
print(d) # 713

# pca=PCA(n_components=154) 
# x2d=pca.fit_transform(x)
# print(x2d.shape) #(70000, 154)

# pca_EVR=pca.explained_variance_ratio_ 
# print(pca_EVR)
# print(sum(pca_EVR)) #0.9499684821687217
