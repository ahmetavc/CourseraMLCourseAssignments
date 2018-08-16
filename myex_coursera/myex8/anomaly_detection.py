import numpy as np 
import matplotlib.pyplot as plt 
from scipy.io import loadmat

mat = loadmat('data/ex8data1.mat')

X = mat['X']



def plotData(X):
	plt.figure(figsize=(8,6))
	plt.plot(X[:,0],X[:,1],'b+')
	plt.xlabel('Latency [ms]',fontsize=16)
	plt.ylabel('Throughput [mb/s]',fontsize=16)
	plt.grid(True)
	

def p(x,u,sigma2):

	m = x.shape[0] # number of examples
	firstTerm = 1/np.sqrt(2*np.pi*sigma2)
	secondTerm = np.exp(-np.square(x-u)/(2*sigma2))
	result = firstTerm*secondTerm

	return np.prod(result,axis=1).reshape(m,-1)

def estimateGaussian(X):

	m = X.shape[0] #number of examples in X
	n = X.shape[1] #number of features

	u = np.sum(X,axis=0)/m #means array n dimensional

	sigma2 = np.sum(np.square(X-u),axis=0)/m

	return u,sigma2

#u and sigma squared by training data
u, sigma2 = estimateGaussian(X)


#this is taken from kaleko, it was hard for me to contour this
def plotContours(mymu, mysigma2):
    delta = .5
    myx = np.arange(0,30,delta)
    myy = np.arange(0,30,delta)
    meshx, meshy = np.meshgrid(myx, myy)
    coord_list = [ entry.ravel() for entry in (meshx, meshy) ]
    points = np.vstack(coord_list).T
    myz = p(points, mymu, mysigma2)
    #if not useMultivariate:
    #    myz = gausOrthog(points, mymu, mysigma2)
    #else: myz = gausMV(points, mymu, mysigma2)
    myz = myz.reshape((myx.shape[0],myx.shape[0]))

    
    
    cont_levels = [10**exp for exp in range(-20,0,3)]
    mycont = plt.contour(meshx, meshy, myz, levels=cont_levels)

    plt.title('Gaussian Contours',fontsize=16)

# First contours without using multivariate gaussian:
# plotData(X)
# useMV = False
# plotContours(*estimateGaussian(X), newFig=False, useMultivariate = useMV)
# plt.show()

#YOU FIND U AND SIGMA_2 FROM TRAINING SET, THESE ARE PARAMETERS THAT YOU GET BY TRAINING YOUR MODEL

ycv = mat['yval']
Xcv = mat['Xval']

pval = p(Xcv,u,sigma2)

def computeF1(predVec, trueVec):
    """
    F1 = 2 * (P*R)/(P+R)
    where P is precision, R is recall
    Precision = "of all predicted y=1, what fraction had true y=1"
    Recall = "of all true y=1, what fraction predicted y=1?
    Note predictionVec and trueLabelVec should be boolean vectors.
    """
    
    P, R = 0., 0.
    if float(np.sum(predVec)):
        P = np.sum([int(trueVec[x]) for x in range(predVec.shape[0]) \
                    if predVec[x]]) / float(np.sum(predVec))
    if float(np.sum(trueVec)):
        R = np.sum([int(predVec[x]) for x in range(trueVec.shape[0]) \
                    if trueVec[x]]) / float(np.sum(trueVec))
        
    return 2*P*R/(P+R) if (P+R) else 0



def selectThreshold(myycv, mypCVs):
    """
    Function to select the best epsilon value from the CV set
    by looping over possible epsilon values and computing the F1
    score for each.
    """
    # Make a list of possible epsilon values
    nsteps = 1000
    epses = np.linspace(np.min(mypCVs),np.max(mypCVs),nsteps)
    
    # Compute the F1 score for each epsilon value, and store the best 
    # F1 score (and corresponding best epsilon)
    bestF1, bestEps = 0, 0
    trueVec = (myycv == 1).flatten()
    for eps in epses:
        predVec = mypCVs < eps
        thisF1 = computeF1(predVec, trueVec)
        if thisF1 > bestF1:
            bestF1 = thisF1
            bestEps = eps
            
    print("Best F1 is %f, best eps is %0.4g."%(bestF1,bestEps))
    return bestF1, bestEps

#bestF1, bestEps = selectThreshold(ycv,pval)

def plotAnomalies(myX, mybestEps):
    ps = p(myX, *estimateGaussian(myX))
    anoms = np.array([myX[x] for x in range(myX.shape[0]) if ps[x] < mybestEps])
   
    plt.scatter(anoms[:,0],anoms[:,1], s=80, facecolors='none', edgecolors='r')

# plotData(X)
# plotContours(u, sigma2)
# plotAnomalies(X, bestEps)
# plt.show()

#HIGH DIMENSIONAL DATASET

mat = loadmat( 'data/ex8data2.mat' )
Xpart2 = mat['X']
ycvpart2 = mat['yval']
Xcvpart2 = mat['Xval']

#find u and sigma2 params
upart2, sigma2part2 = estimateGaussian(Xpart2)

#find best treshold
pval = p(Xcvpart2,upart2,sigma2part2)
bestF1, bestEps = selectThreshold(ycvpart2,pval)

#then see the results
porj = p(Xpart2,upart2,sigma2part2)

#its kind of ridicilious to plot out 7D data
# plotData(Xpart2)
# #plotContours(upart2, sigma2part2)
# plotAnomalies(Xpart2, bestEps)
# plt.show()

anoms = [Xpart2[x] for x in range(Xpart2.shape[0]) if porj[x] < bestEps]
print('# of anomalies found: ',len(anoms))
