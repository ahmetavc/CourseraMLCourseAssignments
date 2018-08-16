import numpy as np 
import matplotlib.pyplot as plt 
from scipy.io import loadmat
import scipy.optimize #fmin_cg to train the linear regression
from sklearn import svm #SVM software

# mat1 = loadmat('data/ex6data1.mat')

# X,y = mat1['X'],mat1['y']

# #svm library will take care of this bias unit
# #X1 = np.insert(X1,0,1,axis=1)


# pos = np.array([X[i] for i in range(X.shape[0]) if y[i] == 1])
# neg = np.array([X[i] for i in range(X.shape[0]) if y[i] == 0])

def plotData(x1,x2):

	plt.figure(figsize=(8,6))
	plt.plot(x1[:,0].flatten(),x1[:,1].flatten(),'k+')
	plt.plot(x2[:,0].flatten(),x2[:,1].flatten(),'yo')
	

#plotData(pos1,neg1)

#Function to draw the SVM boundary
def plotBoundary(my_svm, xmin, xmax, ymin, ymax):
    """
    Function to plot the decision boundary for a trained SVM
    It works by making a grid of x1 ("xvals") and x2 ("yvals") points,
    And for each, computing whether the SVM classifies that point as
    True or False. Then, a contour is drawn with a built-in pyplot function.
    """
    xvals = np.linspace(xmin,xmax,100)
    yvals = np.linspace(ymin,ymax,100)
    zvals = np.zeros((len(xvals),len(yvals)))
    for i in range(len(xvals)):
        for j in range(len(yvals)):
            zvals[i][j] = float(my_svm.predict(np.array([xvals[i],yvals[j]]).reshape(1,2)))


    zvals = zvals.transpose()

    plt.contour( xvals, yvals, zvals, [0])
    plt.title("Decision Boundary")


# Run the SVM training (with C = 1) using SVM software. 
# When C = 1, you should find that the SVM puts the decision boundary 
# in the gap between the two datasets and misclassifies the data point on the far left

#First we make an instance of an SVM with C=1 and 'linear' kernel
# linear_svm = svm.SVC(C=1, kernel='linear')
# # #Now we fit the SVM to our X matrix (no bias unit)
# linear_svm.fit( X, y.flatten() )
# #Now we plot the decision boundary
# plotData(pos,neg)
# plotBoundary(linear_svm,0,4.5,1.5,5)
# plt.show()


#c = 100
# linear_svm = svm.SVC(C=100, kernel='linear')
# linear_svm.fit( X, y.flatten() )
# plotData(pos,neg)
# plotBoundary(linear_svm,0,4.5,1.5,5)
# plt.show()

def gaussKernel(x1,x2,sigma):

	return float(np.exp(-(np.sum(np.square(x1-x2)))/(2*sigma*sigma)))


#print(gaussKernel(np.array([1, 2, 1]),np.array([0, 4, -1]), 2.))

# Now that I've shown I can implement a gaussian Kernel,
# I will use the of-course built-in gaussian kernel in my SVM software
# because it's certainly more optimized than mine.
# It is called 'rbf' and instead of dividing by sigmasquared,
# it multiplies by 'gamma'. As long as I set gamma = sigma^(-2),
# it will work just the same.

#DATASET2

# mat = loadmat('data/ex6data2.mat')

# X,y = mat['X'],mat['y']

# pos = np.array([X[i] for i in range(X.shape[0]) if y[i] == 1])
# neg = np.array([X[i] for i in range(X.shape[0]) if y[i] == 0])

#plotData(pos,neg)

# Train the SVM with the Gaussian kernel on this dataset.

# sigma = 0.1
# gamma = np.power(sigma,-2.)
# gaus_svm = svm.SVC(C=1, kernel='rbf', gamma=gamma)
# gaus_svm.fit( X, y.flatten() )
# plotData(pos,neg)
# plotBoundary(gaus_svm,0,1,.4,1.0)
# plt.show()

#DATASET3

mat = loadmat('data/ex6data3.mat')

X,y,Xval,yval = mat['X'],mat['y'],mat['Xval'],mat['yval']



pos = np.array([X[i] for i in range(X.shape[0]) if y[i] == 1])
neg = np.array([X[i] for i in range(X.shape[0]) if y[i] == 0])

# plotData(pos,neg)
# plt.show()

def modelselection(X,y,Xval,yval):

	steps = [0.01, 0.03, 0.1, 0.3, 1., 3., 10., 30.]
	times = len(steps)

	params = [] #for c and sigma
	score = [] #for max score

	for i in range(times):
		C = steps[i]

		for j in range(times):
			sigma = steps[j]
			params.append((C,sigma))

			gamma = np.power(sigma,-2.)
			gaus_svm = svm.SVC(C=C, kernel='rbf', gamma=gamma)
			gaus_svm.fit( X, y.flatten())

			accuracy = gaus_svm.score(Xval,yval.flatten())
			score.append(accuracy)

	return params[np.array(score).argmax()]


best = modelselection(X,y,Xval,yval)

gaus_svm = svm.SVC(C=best[0], kernel='rbf', gamma = np.power(best[1],-2.))
gaus_svm.fit( X, y.flatten() )
plotData(pos,neg)
plotBoundary(gaus_svm,-.5,.3,-.8,.6)
plt.show()


