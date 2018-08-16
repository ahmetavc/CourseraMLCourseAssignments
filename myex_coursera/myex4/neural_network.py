import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io #Used to load the OCTAVE *.mat files
import scipy.misc #Used to show matrix as an image
import matplotlib.cm as cm #Used to display images in a specific colormap
import random #To pick random images to display
import scipy.optimize #fmin_cg to train neural network
import itertools
from scipy.special import expit #Vectorized sigmoid function

mat = scipy.io.loadmat('ex4data1.mat')
X, y = mat['X'], mat['y']
X = np.insert(X,0,1,axis=1)

#these weigts are got from trained model
weights = scipy.io.loadmat('ex4weights.mat')
theta1, theta2 = weights['Theta1'], weights['Theta2']
#theta1 25x401
#theta2 10x26


#hypothesis
def h(X,theta):
	return np.array(expit(np.dot(X,theta)))


#return all the predictions of last layer units for all examples
def finalresults(X,theta1,theta2):

	a2 = h(X,theta1.T) #resul of second layer
	z3 = np.insert(a2,0,1,axis=1) #input of third layer
	a3 = h(z3,theta2.T) #output of final layer

	#result = (a3.argmax(axis=1) + 1).reshape(-1,1)

	return a3

#predictions = finalresults(X,theta1,theta2) #5000x10

#y is one column example labels
#K is the number of classes
def recodeLabels(y,K):

	labels = np.zeros((y.shape[0],K))

	for i in range(y.shape[0]):

		labels[i][y[i] - 1] = 1

	return labels.T #Kxm - in our example, it is 10X5000


labels = recodeLabels(y,10) #recoded labels


#regularized cost function in neural network
def costFunction(X,theta1,theta2,labels,K,mylambda = 0.):

	m = X.shape[0] # number of examples

	predictions = finalresults(X,theta1,theta2) #5000x10

	costs = np.zeros((m,1))

	for example in range(m):

		term1 = np.dot(np.log(predictions[example,:]),labels[:,example])
		term2 = np.dot(np.log(1 - predictions[example,:]),1 - labels[:,example])
		result = term1 + term2

		costs[example] = result


	t1 = np.sum(np.square(theta1[:,1:]))
	t2 = np.sum(np.square(theta2[:,1:]))

	return (np.sum(costs) / -m) + (mylambda/(2*m))*(t1 + t2)


#print(costFunction(X,theta1,theta2,labels,10,1))


#PART 2 - BACKPROPAGATION

def sigmoidGradient(z):
	return np.array(expit(z)*(1 - expit(z)))


e_init = 0.12

initial_theta1 = np.random.rand(theta1.shape[0],
	theta1.shape[1])*2*e_init - e_init

initial_theta2 = np.random.rand(theta2.shape[0],
	theta2.shape[1])*2*e_init - e_init


#returns activation records z2,a2,z3
def feedforward(X,theta1,theta2):

	#inputs of second layer
	z2 = np.dot(X,theta1.T)

	#activation records of second layer
	a2 = expit(z2)

	a2 = np.insert(a2,0,1,axis=1)

	#inputs of last layer
	z3 = np.dot(a2,theta2.T)

	#output of last layer
	a3 = expit(z3)

	return z2,a2,z3,a3


z2,a2,z3,a3 = feedforward(X,theta1,theta2)

#z2 shape :  (5000, 25)
#a2 shape :  (5000, 26)
#z3 shape :  (5000, 10)
#a3 shape :  (5000, 10)

def backpropagation(X,theta1,theta2,labels,mylambda = 0.):

	m = X.shape[0] # number of the examples

	z2,a2,z3,a3 = feedforward(X,theta1,theta2)

	Delta1 = np.zeros((25,401))
	Delta2 = np.zeros((10,26))

	for t in range(m):

		x_t = X[t,:].reshape(1,401) # x is the single example # (1,401)

		y_t = labels[:,t].reshape(10,1) # labels of the example # (10,1)

		z2_t = z2[t,:].reshape(1,25) # z2 for the example # (1x25)
		a2_t = a2[t,:].reshape(1,26) # a2 for the example # (1x26)
		z3_t = z3[t,:].reshape(1,10) # z3 for the example # (1,10)
		a3_t = a3[t,:].reshape(1,10) # a3 for the example # (1,10)

		delta_3 = np.subtract(a3_t.T,y_t) # (10,1)
		delta_2 = np.multiply(np.dot(theta2.T,delta_3).reshape(26,1)[1:,:],sigmoidGradient(z2_t).reshape(25,1)) # (25x1)


		Delta1 += delta_2.dot(x_t) # 25x1 , 1x401 = 25x401
		Delta2 += delta_3.dot(a2_t) # 10x1 , 1x26 = 10x26


	D1 = Delta1/float(m) #25x401
	D2 = Delta2/float(m) #10x26

	#Regularization:
	D1[:,1:] = D1[:,1:] + (float(mylambda)/m)*theta1[:,1:]
	D2[:,1:] = D2[:,1:] + (float(mylambda)/m)*theta2[:,1:]

	return D1, D2



































