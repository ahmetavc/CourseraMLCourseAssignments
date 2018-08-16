import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy.special import expit


data = np.loadtxt('ex2data1.txt',delimiter=',',usecols=(0,1,2),unpack=True)

X = np.array(data.T[:,0:2])
y = np.array(data.T[:,-1]).reshape(100,1)

m = y.shape[0]
X = np.insert(X,0,1,axis=1)

pos = np.array([X[i] for i in range(X.shape[0]) if y[i] == 1])
neg = np.array([X[i] for i in range(X.shape[0]) if y[i] == 0])

iterations = 1500
alpha = 0.01

def plotData():
	plt.figure(figsize=(8,8))
	plt.plot(pos[:,1],pos[:,2],'k+',label='admitted')
	plt.plot(neg[:,1],neg[:,2],'yo',label='not admitted')
	plt.xlabel('EXAM 1')
	plt.ylabel('EXAM 2')
	boundary_xs = np.array([np.min(X[:,1]), np.max(X[:,1])])
	boundary_ys = (-1./theta[2])*(theta[0] + theta[1]*boundary_xs)
	plt.plot(boundary_xs,boundary_ys,'b-',label='Decision Boundary')
	plt.legend()
	plt.show()



theta = np.array(np.zeros(3)).reshape(3,1) #initial pramaters
initial_theta = np.array(np.zeros(3)).reshape(3,1)


#hypothesis
def h(theta,X):
	return np.array(expit(np.dot(X,theta)))


#cost function for logistic regression
def costFunction(theta,X,y,mylambda = 0.):

	theta = theta.reshape((theta.shape[0],1))

	term1 = np.dot(np.log(h(theta,X).T),y)
	term2 = np.dot(np.log(1-h(theta,X).T),1-y)
	regterm = (mylambda/(2*y.shape[0]))*(np.dot(theta.T[:,1:],theta[1:,:]))
	

	return float(((-1/y.shape[0])*( term1 + term2)) + regterm)


#grdient descent algorithm without regularization
def gradientDescent(theta,X,y,iterations,alpha):

	m = y.shape[0]


	thetahistory = []
	costhistory = []

	for i in range(iterations):

		thetahistory.append(theta[:,0])

		costhistory.append(costFunction(theta,X,y))
		#temp list for thetas

		tmptheta = theta

		for j in range(len(tmptheta)):

			#both lines are correct
			#tmptheta[j] = theta[j] - (alpha/m)*np.sum((h(theta,X) - y)*np.array(X[:,j]).reshape(m,1))	
			tmptheta[j] = tmptheta[j] - (alpha/m)*np.dot((h(theta,X) - y).T, X[:,j].reshape(m,1))

		theta = tmptheta


	return theta, thetahistory, costhistory

#newtheta,thetahistory,costhistory = gradientDescent(theta,X,y,1500,0.001)


def plotConvergence(jvec):
    plt.figure(figsize=(10,6))
    plt.plot(range(len(jvec)),jvec,'bo')
    plt.grid(True)
    plt.title("Convergence of Cost Function")
    plt.xlabel("Iteration number")
    plt.ylabel("Cost function")
    plt.xlim([-0.05*iterations,1.05*iterations])
    #plt.ylim([4,7])
    plt.show()

#plotConvergence(costhistory)

from scipy import optimize

def optimizeTheta(mytheta,myX,myy,mylambda=0.):
    result = optimize.fmin(costFunction, x0=mytheta, args=(myX, myy, mylambda), maxiter=400, full_output=True)
    return result[0], result[1]

#theta, mincost = optimizeTheta(initial_theta,X,y)

#plotData()

def makePrediction(theta,X):

	return h(theta,X) >= 0.5


# pos_correct = float(np.sum(makePrediction(theta,pos)))
# neg_correct = float(np.sum(np.invert(makePrediction(theta,neg))))
# tot = len(pos)+len(neg)
# prcnt_correct = float(pos_correct+neg_correct)/tot
# print("Fraction of training samples correctly predicted: %f." % prcnt_correct)


####
####
## REGULARIZED LOGISTIC REGRESSION


data = np.loadtxt("ex2data2.txt",delimiter=',',usecols=(0,1,2),unpack=True)

data = data.T

X = data[:,0:2] #feature data
y = data[:,2] #label data

m = y.size ##number of examples

y = y.reshape(m,1)

X = np.insert(X,0,1,axis=1)

pos = np.array([X[i] for i in range(m) if y[i] == 1])
neg =np.array([X[i] for i in range(m) if y[i] == 0])


# #PLOTTING DATA
# plt.figure(figsize=(6,6))
# plt.plot(pos[:,1],pos[:,2],'k+',label='y = 1')
# plt.plot(neg[:,1],neg[:,2],'yo',label='y = 0')
# plt.xlabel('Microchip Test 1')
# plt.ylabel('Microchip Test 2')
# plt.legend()
# plt.show()

# def featureMapping(X,limit):

# 	num_feature = X.shape[1] - 1 #number of features

# 	for i in range(1,num_feature + 1):


#I took this method from KALEKO (the MATLAB equivalent was provided in the assignment)
def mapFeature( x1col, x2col ):
    """ 
    Function that takes in a column of n- x1's, a column of n- x2s, and builds
    a n- x 28-dim matrix of featuers as described in the homework assignment
    """
    degrees = 6
    out = np.ones( (x1col.shape[0], 1) )

    for i in range(1, degrees+1):
        for j in range(0, i+1):
            term1 = x1col ** (i-j)
            term2 = x2col ** (j)
            term  = (term1 * term2).reshape( term1.shape[0], 1 ) 
            out   = np.hstack(( out, term ))
    return out


    #Create feature-mapped X matrix
mappedX = mapFeature(X[:,1],X[:,2])


initial_theta = np.zeros((mappedX.shape[1],1))

#gradient descent algorithm with regression
def gradientDescent(theta,X,y,iterations,alpha,mylambda = 0):

	m = y.shape[0]


	thetahistory = []
	costhistory = []

	for i in range(iterations):

		thetahistory.append(theta[:,0])

		costhistory.append(costFunction(theta,X,y,mylambda))
		#temp list for thetas

		tmptheta = theta

		tmptheta[0] = tmptheta[0] - (alpha/m)*np.dot((h(theta,X) - y).T, X[:,0].reshape(m,1))

		for j in range(1,len(tmptheta)):

			#both lines are correct
			#tmptheta[j] = theta[j] - (alpha/m)*np.sum((h(theta,X) - y)*np.array(X[:,j]).reshape(m,1))	
			tmptheta[j] = tmptheta[j]*(1 - (alpha*mylambda/m)) - (alpha/m)*np.dot((h(theta,X) - y).T, X[:,j].reshape(m,1))

		theta = tmptheta

	return theta, thetahistory, costhistory


#newtheta,thetahistory,costhistory = gradientDescent(initial_theta,mappedX,y,1500,0.001,0)

def optimizeRegularizedTheta(mytheta,myX,myy,mylambda=0.):
    result = optimize.minimize(costFunction, mytheta, args=(myX, myy, mylambda),  method='BFGS', options={"maxiter":500, "disp":False} )
    return np.array([result.x]), result.fun
    

initial_theta = np.zeros((mappedX.shape[1],1))  


#theta, mincost = optimizeRegularizedTheta(initial_theta,mappedX,y)

