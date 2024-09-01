import numpy as np
import pandas as pd

filepath = "C:\\Users\\perry\\Downloads\\MNIST_train.csv"

mnist_df = pd.read_csv(filepath)

##check size, shape, and description of data
print(mnist_df.shape)
print(mnist_df.count())
print(mnist_df.describe())