import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io #Used to load the OCTAVE *.mat files
import scipy.misc #Used to show matrix as an image
import matplotlib.cm as cm #Used to display images in a specific colormap
import random #To pick random images to display
from scipy.special import expit #Vectorized sigmoid function
from scipy import optimize

datafile = 'ex3data1.mat'
mat = scipy.io.loadmat( datafile )
X, y = mat['X'], mat['y']

#Insert a column of 1's to X as usual
X = np.insert(X,0,1,axis=1)
#print ("'y' shape: %s. Unique elements in y: %s"%(mat['y'].shape,np.unique(mat['y'])))
#print ("'X' shape: %s. X[0] shape: %s"%(X.shape,X[0].shape))


#hypothesis
def h(theta,X):
	return np.array(expit(np.dot(X,theta)))


#cost function for logistic regression, if lambda is 0 then it is not regularized
def computeCost(theta,X,y,mylambda = 0.):

	theta = theta.reshape((theta.shape[0],1))

	term1 = np.dot(np.log(h(theta,X).T),y)
	term2 = np.dot(np.log(1-h(theta,X).T),1-y)
	regterm = (mylambda/(2*y.shape[0]))*(np.dot(theta.T[:,1:],theta[1:,:]))
	

	return float(((-1/y.shape[0])*( term1 + term2)) + regterm)

# one column theta, for one label
#theta 401x1
#X 5000x401
#y 5000x1
def costGradient(theta,X,y,mytheta = 0.):

	m = y.shape[0] # m = 5000
	hx = h(theta,X) #hx 5000x1
	dif = hx - y #dif 5000x1

	result = (1/m)*np.array(np.dot(X.T,dif)) # 401x1
	regterm = ((mytheta/m)*theta).reshape(theta.shape[0],1) #401x1
	regterm[0] = 0

	result = result + regterm
	return result




#parameter optimizator
#this function minimizes given function
#return necessary parameters of this function for its min val
def optimizeTheta(mytheta,myX,myy,mylambda=0.):
    result = optimize.fmin_cg(computeCost, fprime=costGradient, x0=mytheta, \
                              args=(myX, myy, mylambda), maxiter=50, disp=False,\
                              full_output=True)
    return result[0], result[1]



#one vs All classification
#theta = parameters matris
#X trainign data
#y label data
#K number of label type
def oneVsAll(theta,X,y,K,mylambda = 0.):

	result = np.zeros((K,X.shape[1]))

	for label in range(1,K+1):

		labels = y.copy()

		for i in range(y.size):
			if labels[i] == label:
				labels[i] = 1
			else:
				labels[i] = 0

		optimized_theta = optimizeTheta(theta,X,y,mylambda)

		result[label-1] = optimized_theta

	return result

initial_theta = np.zeros((X.shape[1],1))

#theta = oneVsAll(initial_theta,X,y,10,0)
gradient = costGradient(initial_theta,X,y)

theta = optimizeTheta(initial_theta,X,y) 








