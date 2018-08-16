import numpy as np 
import matplotlib.pyplot as plt
#Import necessary matplotlib tools for 3d plots
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib import cm
import itertools
from numpy.linalg import inv

#FIRST PART, NO VECTORIZATION
#import data
data = pd.read_csv('ex1data1.txt')

##We need to add 1 column for x0
data['constant'] = 1
data['profit'], data['constant'] = data['constant'],data['profit']
data['profit'], data['population'] = data['population'],data['profit']
data.columns = ['constant','population','profit']

values = data.values

X = values[:,0:2] #feature values
y = values[:,2:3]  #results


###Data Visualization
#sns.regplot(x=data["population"], color='r',y=data["profit"], marker='x',fit_reg=False)
#plt.show()

#hyper-parameters
theta0 = 0
theta1 = 0
iterations = 1500
alpha = 0.01


theta = np.array([[0],[0]])

#COST FUNCTION
#X FEATURE DATA MATRIX
#y result matirx
#theta, paramater matrix
def computeCost(X, y, theta):

	number_of_samples = X.shape[0]
	total = 0
	theta_transpose = theta.transpose()

	for i in range(0,number_of_samples):
		x = X[i] # sample, i'th row of X. x is an numpy array now

		h = np.matmul(theta_transpose, x) #hypothesis
		total += (h - y[i])**2 

	return (1/(2*number_of_samples))*total[0]


#GRADIENT DESCENT ALGORITHM
def gradientDescent(X,y,theta,iterations,alpha):


	number_of_samples = X.shape[0]
	theta_transpose = theta.transpose()
	theta_transpose = theta_transpose.astype(np.float32, copy=False)

	costs = [] ##will be used for plotting costs
	thetahistory = [] #Used to visualize the minimization path later on
	total0 = 0 #for theta0
	total1 = 0 #for theta1

	newTheta0 = 0
	newTheta1 =0

	for i in range(0,iterations):

		#print('Step',i)
		#print(computeCost(X,y,theta_transpose.transpose()))
		costs.append(computeCost(X,y,theta_transpose.transpose()))
		thetahistory.append(list(theta_transpose.transpose()[:,0]))

		total0 = 0
		total1 = 0

		for i in range(0,number_of_samples):
			x = X[i] # sample, i'th row of X. x is an numpy array now
			h = np.matmul(theta_transpose, x) #hypothesis
			keep_value = (h - y[i])
			
			total0 += keep_value
			total1 += keep_value*x[1]	

		total0 = (alpha/number_of_samples)*total0
		total1 = (alpha/number_of_samples)*total1


		newTheta0 = float(theta_transpose[0,0] - total0)
		newTheta1 = float(theta_transpose[0,1] - total1)

		
		theta_transpose[0,0] = newTheta0
		theta_transpose[0,1] = newTheta1

	return theta_transpose, costs, thetahistory

#finaltheta,costs,thetahistory = gradientDescent(X,y,theta,iterations,alpha)


#Plot the convergence of the cost function
def plotConvergence(jvec):
    plt.figure(figsize=(10,6))
    plt.plot(range(len(jvec)),jvec,'bo')
    plt.grid(True)
    plt.title("Convergence of Cost Function")
    plt.xlabel("Iteration number")
    plt.ylabel("Cost function")
    #plt.xlim([-0.05*iterations,1.05*iterations])
    #plt.ylim([4,7])
    plt.show()


#plotConvergence(costs)

#Plot the line on top of the data to ensure it looks correct
#this return final hypotesis
def myfit(xval):
    return finaltheta[0][0] + finaltheta[0][1]*xval

#plot method
def plotLine():
	plt.figure(figsize=(10,6))
	plt.plot(X[:,1],y[:,0],'rx',markersize=10,label='Training Data')
	plt.plot(X[:,1],myfit(X[:,1]),'b-',label = 'Hypothesis: h(x) = %0.2f + %0.2fx'%(finaltheta[0][0],finaltheta[0][1]))
	plt.grid(True) #Always plot.grid true!
	plt.ylabel('Profit in $10,000s')
	plt.xlabel('Population of City in 10,000s')
	plt.legend()
	plt.show()


def plot3D():
	fig = plt.figure(figsize=(12,9))
	ax = fig.gca(projection='3d')

	xvals = np.arange(-10,10,.5)
	yvals = np.arange(-1,4,.1)
	myxs, myys, myzs = [], [], []
	for i in xvals:
		for j in yvals:
			myxs.append(i)
			myys.append(j)
			myzs.append(computeCost(X,y,np.array([[i], [j]])))

	scat = ax.scatter(myxs,myys,myzs,c=np.abs(myzs),cmap=plt.get_cmap('YlOrRd'))

	plt.xlabel(r'$\theta_0$',fontsize=30)
	plt.ylabel(r'$\theta_1$',fontsize=30)
	plt.title('Cost (Minimization Path Shown in Blue)',fontsize=30)
	plt.plot([x[0] for x in thetahistory],[x[1] for x in thetahistory],costs,'bo-')
	plt.show()
		

#OPTIONAL PART
# this will be better than first part

data2 = np.loadtxt("ex1data2.txt",delimiter=',',usecols=(0,1,2),unpack=True)

X = np.array(np.transpose(data2[:-1])) # feature matrix
y = np.array(np.transpose(data2[-1:])) # price matrix

m = y.shape[0] #size of examples
X = np.insert(X,0,1,axis=1)

theta = np.zeros((X.shape[1],1))

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

Xnorm = X.copy() #xnorm is normalized feature data

Xnorm, means, stds = featureNormalization(Xnorm)



#hypothesis function, its a (DATAx1) matrix
def h(theta,X):
	return np.dot(X,theta)

def computeCost(theta,X,y):

	return float((1./(2*m)) * np.dot((h(theta,X)-y).T,(h(theta,X)-y)))


def gradientDescent(X,theta):

	thetahistory = [] #will be used for plotting old thetas
	costhistory = [] #will be usef for plotting cost function

	for i in range(iterations):

		costhistory.append(computeCost(theta,X,y))
		thetahistory.append(theta[:,0])

		tmptheta = theta

		for j in range(len(tmptheta)):

			#both lines are correct
			#tmptheta[j] = theta[j] - (alpha/m)*np.sum((h(theta,X) - y)*np.array(X[:,j]).reshape(m,1))	
			tmptheta[j] = tmptheta[j] - (alpha/m)*np.dot((h(theta,X) - y).T, X[:,j].reshape(m,1))

		theta = tmptheta

	return theta, thetahistory, costhistory

initial_theta = np.zeros((Xnorm.shape[1],1))

theta, thetahistory, costhistory = gradientDescent(Xnorm,initial_theta)

plotConvergence(costhistory)

def normal(X,y):

	return np.dot(np.dot(inv(np.dot(X.T,X)),X.T),y)

#print("Normal equation prediction for price of house with 1650 square feet and 3 bedrooms")
#print("$%0.2f" % float(h(normal(X,y),[1,1650.,3])))
#[[340412.65957444]
 #[109447.6963442 ]
 #[ -6578.25494988]]

#print "Final result theta parameters: \n",theta
#print("Check of result: What is price of house with 1650 square feet and 3 bedrooms?")
#ytestscaled = [1,(1650.-means[0])/stds[0],(3.- means[1])/stds[1]]
#print( "$%0.2f" % float(h(theta,ytestscaled)))