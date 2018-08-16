import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat #Used to load the OCTAVE *.mat files
from random import sample #Used for random initialization
import scipy.misc #Used to show matrix as an image
import matplotlib.cm as cm #Used to display images in a specific colormap
from scipy import linalg #Used for the "SVD" function


mat = loadmat('data/ex7data2.mat')

X = mat['X']

#Visualizing the data
def plotData(myX,mycentroids,myidxs = None):

    """
    Fucntion to plot the data and color it accordingly.
    myidxs should be the latest iteraction index vector
    mycentroids should be a vector of centroids, one per iteration
    """
    
    colors = ['b','g','gold','darkorange','salmon','olivedrab']
    
    assert myX[0].shape == mycentroids[0][0].shape
    assert mycentroids[-1].shape[0] <= len(colors)

    #If idxs is supplied, divide up X into colors
    if myidxs is not None:
        assert myidxs.shape[0] == myX.shape[0]
        subX = []
        for x in range(mycentroids[0].shape[0]):
            subX.append(np.array([myX[i] for i in range(myX.shape[0]) if myidxs[i] == x]))
    else:
        subX = [myX]
        
    fig = plt.figure(figsize=(7,5))
    for x in range(len(subX)):
        newX = subX[x]
        plt.plot(newX[:,0],newX[:,1],'o',color=colors[x],
                 alpha=0.75, label='Data Points: Cluster %d'%x)
    plt.xlabel('x1',fontsize=14)
    plt.ylabel('x2',fontsize=14)
    plt.title('Plot of X Points',fontsize=16)
    plt.grid(True)

    #Drawing a history of centroid movement
    tempx, tempy = [], []
    for mycentroid in mycentroids:
        tempx.append(mycentroid[:,0])
        tempy.append(mycentroid[:,1])
    
    for x in range(len(tempx[0])):
        plt.plot(tempx, tempy, 'rx--', markersize=8)

    leg = plt.legend(loc=4, framealpha=0.5)
    plt.show()

def findClosestCentroids(X,centroids):

	m = X.shape[0] #number of examples

	idx = []

	for ex in range(m):

		temp = centroids
		temp = temp - X[ex,:] # u1-x1, u2-x2
		temp = np.square(temp) # (u1-x1)ˆ2, (u2-x2)ˆ2
		temp = np.sum(temp,axis=1) # (u1-x1)ˆ2 + (u2-x2)ˆ2
		temp = np.sqrt(temp) # [(u1-x1)ˆ2 + (u2-x2)ˆ2]ˆ1/2 
		#so this is the distance between x and all the centroids

		idx.append(temp.argmin())


	return np.array(idx)

K = 3
#Choose the initial centroids matching ex7.m assignment script
initial_centroids = np.array([[3,3],[6,2],[8,5]])

#MY IMPLEMENTATION, NOT MUCH EFFICIENT AND GOT A BUG WHEN A VECTOR HAS NO CLOSEST POINT
# def computeCentroids(X,idx,K):


# 	centroids = np.zeros((K,X.shape[1]))

# 	#this will keep the number of instances of one vector
# 	count = np.zeros((K,1)) 

# 	for i in range(X.shape[0]):

# 		centroids[idx[i]] += X[i]
# 		count[idx[i]] += 1

# 	for i in range(centroids.shape[0]):

# 		if count[i] == 0:
# 			centroids = np.delete(centroids,centroids[i],axis=0)
# 			i -= 1
# 			continue

# 		centroids[i] /= count[i]

# 	return centroids

def computeCentroids(myX, myidxs):
    """
    Function takes in the X matrix and the index vector
    and computes a new centroid matrix.
    """
    subX = []
    for x in range(len(np.unique(myidxs))):
        subX.append(np.array([myX[i] for i in range(myX.shape[0]) if myidxs[i] == x]))
    return np.array([np.mean(thisX,axis=0) for thisX in subX])


def runKMeans(X, initial_centroids, K, n_iter):

	centroid_history = []
	current_centroids = initial_centroids

	for i in range(n_iter):
		centroid_history.append(current_centroids)
		idx = findClosestCentroids(X,current_centroids)
		current_centroids = computeCentroids(X, idx)

	return idx, centroid_history


#idxs, centroid_history = runKMeans(X,initial_centroids,K=3,n_iter=10)

def chooseKRandomCentroids(myX, K):
    rand_indices = np.random.choice(range(0,myX.shape[0]),K)
    return np.array([myX[i] for i in rand_indices])



# This creates a three-dimensional matrix A whose first two indices 
# identify a pixel position and whose last index represents red, green, or blue.
A = scipy.misc.imread('data/bird_small.png')
#A shape is  (128, 128, 3)


# Divide every entry in A by 255 so all values are in the range of 0 to 1
A = A / 255. 

# Unroll the image to shape (16384,3) (16384 is 128*128)
A = A.reshape(-1, 3)

myK = 16
idxs, centroid_history = runKMeans(A,chooseKRandomCentroids(A,myK),
                                   myK,n_iter=10)

# Now I have 16 centroids, each representing a color.
# Let's assign an index to each pixel in the original image dictating
# which of the 16 colors it should be
idxs = findClosestCentroids(A, centroid_history[-1])

final_centroids = centroid_history[-1]
# Now loop through the original image and form a new image
# that only has 16 colors in it
final_image = np.zeros((idxs.shape[0],3))
for x in range(final_image.shape[0]):
    final_image[x] = final_centroids[int(idxs[x])]


# Reshape the original image and the new, final image and draw them
# To see what the "compressed" image looks like
plt.figure()
dummy = plt.imshow(A.reshape(128,128,3))
plt.figure()
dummy = plt.imshow(final_image.reshape(128,128,3))
plt.show()









