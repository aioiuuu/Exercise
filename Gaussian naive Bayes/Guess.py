from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
x,y=make_blobs(n_samples=100,centers=2,random_state=0,cluster_std=1.0)
plt.scatter(x[:,0],x[:,1],c=y,s=50,cmap='winter')
