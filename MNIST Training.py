import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn

filepath = "C:\\Users\\perry\\Downloads\\MNIST_train.csv"

mnist_df = pd.read_csv(filepath)

##check size, shape, and description of data
print(mnist_df.shape)
print(mnist_df.count())
print(mnist_df.describe())

#Transform the data to an array
X = mnist_df.to_numpy()

#split data into x,y
y = X[:, 0]
X = X[:, 1:]

print(y.shape)
print(X.shape)

##building non-naive GB Classifer
class GaussBayes():
    
    def fit(self,X_train,y_train,epsilon=1e-3):
        self.likelihoods = dict()
        self.priors = dict()
        self.K = set(y_train.astype(int))
        
        for k in self.K:
            X_train_k = X_train[y_train==k, :]
            N_k, D = X_train_k.shape
            mu_k = X_train_k.mean(axis = 0)
            cov_k = (1 / (N_k-1))*np.matmul((X_train_k - mu_k).T,X_train_k-mu_k)+epsilon*np.identity(D)
            
            self.likelihoods[k] = {"mean":mu_k, "cov":cov_k}
            self.priors[k] = len(X_train_k) / len(X_train)
            
    def predict(self, X_train):
        N, D = X_train.shape
        P_hat = np.zeros((N, len(self.K)))
        
        #make predictions within loop
        for k, l in self.likelihoods.items():
            P_hat[:, k] = mvn.logpdf(X_train, l["mean"], l["cov"])+np.log(self.priors[k])
        
        return P_hat.argmax(axis =1)
    
    def accuracy(y_train, y_train_hat):
        return np.mean(y_train == y_train_hat)
            