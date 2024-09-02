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