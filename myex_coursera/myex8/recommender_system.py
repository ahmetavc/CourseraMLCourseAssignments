import numpy as np 
import matplotlib.pyplot as plt 
from scipy.io import loadmat
import scipy.optimize

mat = loadmat('data/ex8_movies.mat')

R,Y = mat['R'],mat['Y']

mat = loadmat('data/ex8_movieParams.mat')
X = mat['X']
Theta = mat['Theta']
nu = int(mat['num_users'])
nm = int(mat['num_movies'])
nf = int(mat['num_features'])

# The "parameters" we are minimizing are both the elements of the
# X matrix (nm*nf) and of the Theta matrix (nu*nf)
# To use off-the-shelf minimizers we need to flatten these matrices
# into one long array
def flattenParams(myX, myTheta):
    """
    Hand this function an X matrix and a Theta matrix and it will flatten
    it into into one long (nm*nf + nu*nf,1) shaped numpy array
    """
    return np.concatenate((myX.flatten(),myTheta.flatten()))

# A utility function to re-shape the X and Theta will probably come in handy
def reshapeParams(flattened_XandTheta, mynm, mynu, mynf):
    assert flattened_XandTheta.shape[0] == int(nm*nf+nu*nf)
    
    reX = flattened_XandTheta[:int(mynm*mynf)].reshape((mynm,mynf))
    reTheta = flattened_XandTheta[int(mynm*mynf):].reshape((mynu,mynf))
    
    return reX, reTheta


#in order to use optimize function, we need to give theta and X as one flattened parameter
def cofiCostFunc(params, Y, R, nu, nm, nf, mylambda = 0.):

	# Unfold the X and Theta matrices from the flattened params
    myX, myTheta = reshapeParams(params, nm, nu, nf)

    pre = np.dot(myX,myTheta.T) #HOLE PREDICTIONS 1682X943
    #some of rates in this matrix are null because there are gaps in evaluation matrix
    #by muplitplying pre with R, we make 0 all unrated elements

    pre = np.multiply(pre,R)
    	
    result = 0.5 * np.sum(np.square(pre - Y))

    ##regularization
    result += (mylambda/2.) * np.sum(np.square(myTheta))
    result += (mylambda/2.) * np.sum(np.square(myX))

    return result


#For now, reduce the data set size so that this runs faster
# nu = 4; nm = 5; nf = 3
# X = X[:nm,:nf]
# Theta = Theta[:nu,:nf]
# Y = Y[:nm,:nu]
# R = R[:nm,:nu]

#print(cofiCostFunc(params, Y, R, nu, nm, nf,mylambda =1.5))


# Remember: use the exact same input arguments for gradient function
# as for the cost function (the off-the-shelf minimizer requires this)
def cofiGrad(params, Y, R, nu, nm, nf, mylambda = 0.):

	# Unfold the X and Theta matrices from the flattened params
    myX, myTheta = reshapeParams(params, nm, nu, nf)

    term = np.dot(myX,myTheta.T) #HOLE PREDICTIONS 1682X943
    #some of rates in this matrix are null because there are gaps in evaluation matrix
    #by muplitplying term with R, we make 0 all unrated elements

    term = np.multiply(term,R) 
    term = term - Y

    # Lastly dot this with Theta such that the resulting matrix has the
    # same shape as the X matrix
    Xgrad = term.dot(myTheta)
    
    # Now the Theta gradient term (reusing the "term1" variable)
    Thetagrad = term.T.dot(myX)

    # Regularization stuff
    Xgrad += mylambda * myX
    Thetagrad += mylambda * myTheta

    return flattenParams(Xgrad, Thetagrad)


#Let's check my gradient computation real quick:
def checkGradient(myparams, myY, myR, mynu, mynm, mynf, mylambda = 0.):
    
    print ('Numerical Gradient \t cofiGrad \t\t Difference')
    
    # Compute a numerical gradient with an epsilon perturbation vector
    myeps = 0.0001
    nparams = len(myparams)
    epsvec = np.zeros(nparams)
    # These are my implemented gradient solutions
    mygrads = cofiGrad(myparams,myY,myR,mynu,mynm,mynf,mylambda)

    # Choose 10 random elements of my combined (X, Theta) param vector
    # and compute the numerical gradient for each... print to screen
    # the numerical gradient next to the my cofiGradient to inspect
    
    for i in range(10):
        idx = np.random.randint(0,nparams)
        epsvec[idx] = myeps
        loss1 = cofiCostFunc(myparams-epsvec,myY,myR,mynu,mynm,mynf,mylambda)
        loss2 = cofiCostFunc(myparams+epsvec,myY,myR,mynu,mynm,mynf,mylambda)
        mygrad = (loss2 - loss1) / (2*myeps)
        epsvec[idx] = 0
        print ('%0.15f \t %0.15f \t %0.15f' % \
        (mygrad, mygrads[idx],mygrad - mygrads[idx]))


# print ("Checking gradient with lambda = 0...")
# checkGradient(flattenParams(X,Theta),Y,R,nu,nm,nf)
# print ("\nChecking gradient with lambda = 1.5...")
# checkGradient(flattenParams(X,Theta),Y,R,nu,nm,nf,mylambda = 1.5)

#return original data
mat = loadmat('data/ex8_movies.mat')

R,Y = mat['R'],mat['Y']

mat = loadmat('data/ex8_movieParams.mat')
X = mat['X']
Theta = mat['Theta']
nu = int(mat['num_users'])
nm = int(mat['num_movies'])
nf = int(mat['num_features'])

######################
######################
######################
######################

#MAKE YOUR RATINGS
my_ratings = np.zeros((1682,1))
my_ratings[0]   = 4
my_ratings[97]  = 2
my_ratings[6]   = 3
my_ratings[11]  = 5
my_ratings[53]  = 4
my_ratings[63]  = 5
my_ratings[65]  = 3
my_ratings[68]  = 5
my_ratings[182] = 4
my_ratings[225] = 5
my_ratings[354] = 5


##ADD YOUR RATINGS TO THE Y MATRIX AND R
myR_row = my_ratings > 0
Y = np.hstack((Y,my_ratings))
R = np.hstack((R,myR_row))
nm, nu = Y.shape
######################
######################
######################
######################



#NORMALIZATION FOR NEW USER
def normalizeRatings(myY, myR):
    """
    Preprocess data by subtracting mean rating for every movie (every row)
    This is important because without this, a user who hasn't rated any movies
    will have a predicted score of 0 for every movie, when in reality
    they should have a predicted score of [average score of that movie].
    """

    # The mean is only counting movies that were rated
    Ymean = np.sum(myY,axis=1)/np.sum(myR,axis=1)
    Ymean = Ymean.reshape((Ymean.shape[0],1))
    
    return myY-Ymean, Ymean


Ynorm, Ymean = normalizeRatings(Y,R)


# Generate random initial parameters, Theta and X
X = np.random.rand(nm,nf)
Theta = np.random.rand(nu,nf)
myflat = flattenParams(X, Theta)

# Regularization parameter of 10 is used (as used in the homework assignment)
mylambda = 10.

# Training the actual model with fmin_cg
#train accoirding to normalized Y, but optimize problem does not allow me
#as a result, If I add mean to the rows, I end up 5-10 rank system
#if I did not add means, I cannot propose anything to new users
result = scipy.optimize.fmin_cg(cofiCostFunc, x0=myflat, fprime=cofiGrad, \
                               args=(Y,R,nu,nm,nf,mylambda), \
                                maxiter=50,disp=True,full_output=True)


#fit parameters
Xfit, Thetafit = reshapeParams(result[0], nm, nu, nf)

# After training the model, now make recommendations by computing
# the predictions matrix
#this method fills all the empty movie rates for all users,
#according to these filled blanks, we will recommend users movies that they havent watch yet
def predict(Xfit,Thetafit,Ymean):

	prediction_matrix = Xfit.dot(Thetafit.T)

	#adding ymean for probable new user
	return prediction_matrix #+ Ymean


prediction_matrix = predict(Xfit,Thetafit,Ymean)
#print(prediction_matrix)
my_predictions = prediction_matrix[:,-1]


# Sort my predictions from highest to lowest
pred_idxs_sorted = np.argsort(my_predictions)
pred_idxs_sorted[:] = pred_idxs_sorted[::-1]

print ("Top recommendations for you:")
for i in range(10):
    print (my_predictions[pred_idxs_sorted[i]])
    
print ("\nOriginal ratings provided:")
for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print (my_ratings[i])









