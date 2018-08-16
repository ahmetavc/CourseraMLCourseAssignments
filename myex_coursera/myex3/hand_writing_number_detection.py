import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# load MATLAB files
from scipy.io import loadmat
from scipy.optimize import minimize

from sklearn.linear_model import LogisticRegression
 

data = loadmat('ex3data1.mat')

y= data['y']
# Add constant for intercept
X = np.c_[np.ones((data['X'].shape[0],1)), data['X']]


def sigmoid(z):
    return(1 / (1 + np.exp(-z)))



def lrcostFunctionReg(theta, reg, X, y):
    m = y.size
    h = sigmoid(X.dot(theta))
    
    J = -1*(1/m)*(np.log(h).T.dot(y)+np.log(1-h).T.dot(1-y)) + (reg/(2*m))*np.sum(np.square(theta[1:]))
    
    if np.isnan(J[0]):
        return(np.inf)
    return(J[0])


def lrgradientReg(theta, reg, X,y):
    m = y.size
    h = sigmoid(X.dot(theta.reshape(-1,1)))
      
    grad = (1/m)*X.T.dot(h-y) + (reg/m)*np.r_[[[0]],theta[1:].reshape(-1,1)]
        
    return(grad.flatten())

def oneVsAll(features, classes, n_labels, reg):
    initial_theta = np.zeros((X.shape[1],1))  # 401x1
    all_theta = np.zeros((n_labels, X.shape[1])) #10x401

    for c in np.arange(1, n_labels+1):
        res = minimize(lrcostFunctionReg, initial_theta, args=(reg, features, (classes == c)*1), method=None,
                       jac=lrgradientReg, options={'maxiter':50})
        all_theta[c-1] = res.x
    return(all_theta)


#theta = oneVsAll(X, y, 10, 0)
def oneVsAllpredict(X,theta):

	prediction = theta.dot(X.T)
	result = (prediction.argmax(axis=0).T.reshape(-1,1) + 1)

	return result

def predictionResults(pre,y):

	true = 0
	m = y.shape[0]

	for i in range(m):
		if pre[i] == y[i]:
			true += 1

	return (true/m)*100

#predictions = oneVsAllpredict(X,theta)
#result = predictionResults(predictions,y)

##PART 2 NEURAL NETWORK
weights = loadmat('ex3weights.mat')
theta1, theta2 = weights['Theta1'], weights['Theta2']

def neuralNetwork(X,y,theta1,theta2):

	first = sigmoid(np.dot(X,theta1.T))

	first = np.insert(first,0,1,axis=1) #add bais unit

	second = sigmoid(np.dot(first,theta2.T))

	result = (second.argmax(axis=1) + 1).reshape(-1,1)

	return result

prediction_nn = neuralNetwork(X,y,theta1,theta2)

result = predictionResults(prediction_nn,y)

print(result)






