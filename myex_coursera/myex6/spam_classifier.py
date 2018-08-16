import numpy as np 
import matplotlib.pyplot as plt 
from scipy.io import loadmat
import scipy.optimize #fmin_cg to train the linear regression
from sklearn import svm #SVM software
import re

import nltk, nltk.stem.porter



def preProcess(email):

	#make all letters in the email lower case
	email = email.lower()

	#strip html tags
	email = re.sub('<[^<>]+>', ' ', email)

	#Any numbers get replaced with the string 'number'
	email = re.sub('[0-9]+', 'number', email)

	#Anything starting with http or https:// replaced with 'httpaddr'
	email = re.sub('(http|https)://[^\s]*', 'httpaddr', email)


	#Strings with "@" in the middle are considered emails --> 'emailaddr'
	email = re.sub('[^\s]+@[^\s]+', 'emailaddr', email);

	#The '$' sign gets replaced with 'dollar'
	email = re.sub('[$]+', 'dollar', email);

	return email



def emailtoTokenList( raw_email ):
    """
    Function that takes in preprocessed (simplified) email, tokenizes it,
    stems each word, and returns an (ordered) list of tokens in the e-mail
    """
    
    # I'll use the NLTK stemmer because it more accurately duplicates the
    # performance of the OCTAVE implementation in the assignment
    stemmer = nltk.stem.porter.PorterStemmer()
    
    email = preProcess(raw_email)

    #Split the e-mail into individual words (tokens) (split by the delimiter ' ')
    #but also split by delimiters '@', '$', '/', etc etc
    #Splitting by many delimiters is easiest with re.split()
    tokens = re.split('[ \@\$\/\#\.\-\:\&\*\+\=\[\]\?\!\(\)\{\}\,\'\"\>\_\<\;\%]', email)

    tokenlist = []
    for token in tokens:
        
        #Remove any non alphanumeric characters
        token = re.sub('[^a-zA-Z0-9]', '', token);

        #Use the Porter stemmer to stem the word
        stemmed = stemmer.stem( token )
        
        #Throw out empty tokens
        if not len(token): continue
            
        #Store a list of all unique stemmed words
        tokenlist.append(stemmed)
            
    return tokenlist


def vocabToDic(reverse=False):

	vocab_dict = {}
 	
 	#substractin 1 because in phyton, starting index is 1
	with open("vocab.txt") as f:
		for line in f:
			(val, key) = line.split()
			if not reverse:
				vocab_dict[key] = int(val) - 1
			else:
				vocab_dict[int(val)-1] = key
                
	return vocab_dict


def wordIndices(raw_email,vocab_dict):

	tokens = emailtoTokenList(raw_email)

	index_list = [ vocab_dict[token] for token in tokens if token in vocab_dict ]
	return index_list



def featureVector(email,vocab):

	indices = wordIndices(email,vocab)

	vector = np.zeros((len(vocab),1))

	for index in indices:
		vector[index] = 1

	return vector





# vocab_dict = vocabToDic()
# email_contents = open( 'emailSample1.txt', 'r' ).read()
# test_fv = featureVector(email_contents,vocab_dict)
# print("Length of feature vector is %d" % len(test_fv))
# print("Number of non-zero entries is: %d" % sum(test_fv==1))


# Training set
datafile = 'data/spamTrain.mat'
mat = loadmat( datafile )
X, y = mat['X'], mat['y']
#NOT inserting a column of 1's in case SVM software does it for me automatically...
#X =     np.insert(X    ,0,1,axis=1)

# Test set
datafile = 'data/spamTest.mat'
mat = loadmat( datafile )
Xtest, ytest = mat['Xtest'], mat['ytest']

pos = np.array([X[i] for i in range(X.shape[0]) if y[i] == 1])
neg = np.array([X[i] for i in range(X.shape[0]) if y[i] == 0])

# print ('Total number of training emails = ',X.shape[0])
# print ('Number of training spam emails = ',pos.shape[0])
# print ('Number of training nonspam emails = ',neg.shape[0])




linear_svm = svm.SVC(C=0.1, kernel='linear')

# Now we fit the SVM to our X matrix, given the labels y
linear_svm.fit( X, y.flatten() )

#print('Train Accuracy: ',linear_svm.score(X, y.flatten()))
#print('Test Accuracy: ',linear_svm.score(Xtest, ytest.flatten()))

#inspect the parameters to see which words the classifier thinks are the most predictive of spam.
def topPredictor(svm,vocab):

	sorted_indices = np.argsort( svm.coef_, axis=None )[::-1]
	liste = []

	for i in range(15):
		liste.append(vocab[sorted_indices[i]])

	return liste


vocab_dict = vocabToDic(reverse = True)
print(topPredictor(linear_svm,vocab_dict))




