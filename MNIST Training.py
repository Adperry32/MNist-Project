import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn

filepath = "C:\\Users\\perry\\Downloads\\MNIST_train.csv"
file_test ="C:\\Users\\perry\\Downloads\\MNIST_test.csv"

mnist_df = pd.read_csv(filepath)
mnist_test_df = pd.read_csv(file_test)

##check size, shape, and description of data
print(mnist_df.shape)
print(mnist_df.count())
print(mnist_df.describe())

#Transform the data to an array
X = mnist_df.to_numpy()
X_test = mnist_test_df.to_numpy()


#split data into x,y
y = X[:, 2]
X = X[:, 3:]

#split test data into x,y
y_test = X_test[:, 2]
X_test = X_test[:, 3:]

print(y.shape)
print(X.shape)
#print(y_test.shape)
#print(X_test.shape)

#needed to normalize the data to [0, 1] since there is an image.
X = X / 255.0
X_test = X_test / 255.0


#function to check accuracy of models performance
def accuracy(y, y_hat):
    return np.mean(y == y_hat)

    #building non-naive GB Classifer
class GaussBayes():
    
    def fit(self,X,y,epsilon=1e-2):
        self.likelihoods = dict()
        self.priors = dict()
        self.K = set(y.astype(int))
        
        for k in self.K:
            X_k = X[y==k, :]
            N_k, D = X_k.shape
            mu_k = X_k.mean(axis = 0)
            cov_k = (1 / (N_k-1))*np.matmul((X_k - mu_k).T,X_k-mu_k)+epsilon*np.identity(D)
            
            self.likelihoods[k] = {"mean":mu_k, "cov":cov_k}
            self.priors[k] = len(X_k) / len(X)
            
    def predict(self, X):
        N, D = X.shape
        P_hat = np.zeros((N, len(self.K)))
        
        #make predictions within loop
        for k, l in self.likelihoods.items():
            P_hat[:, k] = mvn.logpdf(X, l["mean"], l["cov"])+np.log(self.priors[k])
        
        return P_hat.argmax(axis =1)
    

#building of KNN classifier to check accuracy verse the GB  
class KNN():
    def fit(self, X,y):
        self.X=X
        self.y=y
        
    def predict(self, X, K, epsilon=1e-2):
        N = len(X)
        y_hat = np.zeros(N)
        
        for i in range(N):
            dist = np.sum((self.X-X[i])**2, axis =1)
            index = np.argsort(dist)[:K]
            gamma_k = 1/(np.sqrt(dist[index]+epsilon))
            y_hat = np.bincount(self.y[index],weights = gamma_k).argmax()
        
        return y_hat
    
#instantiate the classes 
gnb = GaussBayes()
knn = KNN()

gnb.fit(X, y)
knn.fit(X, y)

#y_PredKNN = knn.predict(X_test, 20)
#y_KNN_hat = knn.predict(X, 20)

y_hat = gnb.predict(X)
y_Pred = gnb.predict(X_test)

#variables to hold accuracy score 
train_accuracy = accuracy(y, y_hat)
test_accuracy = accuracy(y_test, y_Pred)


#print out individual scores
print(f"Training Accuracy GnB: {train_accuracy * 100:.2f}%")
print(f"Test Accuracy GnB: {test_accuracy * 100:.2f}%")
#print(f"Training Accuracy KNN: {y_KNN_hat * 100:.2f}%")
#print (f"Test Accuracy KNN: {y_PredKNN * 100:.2f}%")

#present a demo of both the actual data and predicted data from the model side by side
demo = np.random.choice(X_test.shape[0], 5 , replace = False)
demo_images = X_test[demo]
demo_labels = y_test[demo]

p_Hat_labels = gnb.predict(demo_images)


#loop to display images and predictions 
for i in range(5):
    plt.imshow(demo_images[i].reshape(28,28), cmap='gray')
    plt.title(f'Actual: {int(demo_labels[i])}, Predicted: {p_Hat_labels[i]}')
    plt.axis('off')
    plt.show()
