import numpy as np
import pandas as pd

filepath = "C:\\Users\\perry\\Downloads\\MNIST_train.csv"

mnist_df = pd.read_csv(filepath)

##check size, shape, and description of data
print(mnist_df.shape)
print(mnist_df.count())
print(mnist_df.describe())

#Transform the data to an array
X = mnist_df.to_numpy()

#split data into x,y
y = X[:, -1]
X = X[:,:-1]

print(y.shape)
print(X.shape)