import numpy as np 
import matplotlib.pyplot as plt 
from scipy.io import loadmat
import scipy.optimize

mat = loadmat('ex5data1.mat')

X, y = mat['X'], mat['y']
#Cross validation set
Xval, yval = mat['Xval'], mat['yval']
#Test set
Xtest, ytest = mat['Xtest'], mat['ytest']

X = np.insert(X,0,1,axis=1)
Xval = np.insert(Xval,0,1,axis=1)
Xtest = np.insert(Xtest,0,1,axis=1)

m = X.shape[0] #number of training examples
m_cv = Xval.shape[0] #number of cv examples
m_test = Xtest.shape[0] #number of cv examples


##plotting the training data
def plotData():
	xmin, xmax = -80,80
	ymin, ymax = -60,40
	plt.figure(figsize=(8,5))
	plt.ylabel('Water flowing out of the dam (y)')
	plt.xlabel('Change in water level (x)')
	plt.plot(X[:,1],y,'rx')
	#plt.xlim([xmin,xmax])
	#plt.ylim([ymin,ymax])
	plt.grid(True)
    
#plotData()

#hypothesis
def h(X,theta):
	return np.array(np.dot(X,theta))


def costFunction(theta,X,y,mylambda = 0.):
	theta = theta.reshape(-1,1)
	term1 = float((np.sum(np.square(h(X,theta)-y)))/(2*m))
	term2 = float(np.sum((np.square(theta[1:,:])))*(mylambda/(2*m)))

	return (term1 + term2)

def gradient(theta,X,y,mylambda = 0.):

	theta = theta.reshape(-1,1)

	grad = (1/float(m))*X.T.dot(h(X,theta)-y) #for theta0
	reg =  float((mylambda/m))*theta
	reg[0] = 0

	reg.reshape((grad.shape[0],1))

	return np.array(grad + reg).flatten()

def optimizeTheta(myTheta_initial, myX, myy, mylambda=0.,print_output=True):
    fit_theta = scipy.optimize.fmin_cg(costFunction,x0=myTheta_initial,\
                                       fprime=gradient,\
                                       args=(myX,myy,mylambda),\
                                       disp=print_output,\
                                       epsilon=1.49e-12,\
                                       maxiter=1000)
    fit_theta = fit_theta.reshape((myTheta_initial.shape[0],1))
    return fit_theta



#fit_theta = optimizeTheta(initial_theta,X,y,0.)

#plotData()
#plt.plot(X[:,1],h(X,fit_theta).flatten())
#plt.show()


def learningCurves(X,y,Xval,yval,mylambda = 0.):

	initial_theta = np.ones((X.shape[1],1))
	trainError = []
	cvError = []
	ex = []
	print(Xval.shape)

	for i in range(2,m+1):
		ex.append(i) #adding number of examples for plottin

		theta = optimizeTheta(initial_theta,X[:i],y[:i],mylambda,False)

		t_error = costFunction(theta,X[:i],y[:i],mylambda)
		trainError.append(t_error)

		cv_error = costFunction(theta,Xval,yval,mylambda)
		cvError.append(cv_error)


	plt.figure(figsize=(8,5))
	plt.ylabel('Error')
	plt.xlabel('Number of training examples')
	plt.plot(ex,trainError,label='Train')
	plt.plot(ex,cvError,label='Cross validation')
	plt.grid(True)
	plt.legend()
	plt.show()

#learningCurves(X,y,Xval,yval)

def mapFeature(X,p):

	pos = X.shape[1]

	for i in range(2,p+1):
		X = np.insert(X,pos,X[:,1]**i,axis=1)

	return X


X_mapped = mapFeature(X,8)


def featureNormalization(X):

	column_number = X.shape[1] # how many features

	means = [] #used for keeping mean values of feature datas
	stds = [] #used for keepin std values of feature datas

	for i in range(1,column_number):

		mean_feature = np.mean(X[:,i])
		means.append(mean_feature)

		std_feature = np.std(X[:,i])
		stds.append(std_feature)

		X[:,i] = (X[:,i]-mean_feature)/std_feature

	return X,means,stds


X_mapped,means,stds = featureNormalization(X_mapped)

initial_theta = np.ones((X_mapped.shape[1],1))

fit_theta = optimizeTheta(initial_theta,X_mapped,y,1)

#PLOTTING THE MODEL AND DATA

# plotData()
# idx = np.argsort(X[:,1])
# xs = np.array(X[:,1])[idx]
# ys = np.array(h(X_mapped,fit_theta).flatten())[idx]
# plt.plot(xs,ys,'b--')
# plt.show()


Xval_mapped = mapFeature(Xval,8)
Xval_mapped = featureNormalization(Xval_mapped)[0]

#learningCurves(X_mapped,y,Xval_mapped,yval,mylambda = 1)


def bestAlpha(Xval,yval,fit_theta):

	alphas = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
	errors = {}

	for i in alphas:

		errors[i] =  costFunction(fit_theta,Xval,yval,i)

	return min(errors, key=errors.get) #return min value's key

print(bestAlpha(Xval_mapped,yval,fit_theta))
