import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib import cm

def computeCost(X, y, theta):
    m = y.size
    XthetaMinusY = np.dot(X, theta) - y
    
    # Equation:
    # (1/(2*m))*(X*theta - y)'*(X*theta - y)
    equation = np.dot((1/(2*m)), np.dot(np.transpose(XthetaMinusY), (XthetaMinusY)))
    return equation[0][0]


# returns: theta( arr(1 x 2) ), errorHistory( arr(numberOfIterations x 1) )
#
# X -> [1, population(1)]   Y -> [Profit(1)]    alpha -> learningRate
#      [1, population(2)]        [Profit(2)]    
#      ...                       ...
#      [1, population(n)]        [Profit(n)]
def gradientDescent(X, y, alpha, theta, numberOfIterations):
    m = y.size
    errorHistory = np.zeros((numberOfIterations, 1))

    for i in range(0, numberOfIterations):
        # theta = theta - (alpha/m)*X'*(X*theta - y)
        # where:
        # X' - transpose of X
        theta = theta - (alpha/m)*np.dot(np.transpose(X), (np.dot(X,theta) - y))
        errorHistory[i] = computeCost(X, y, theta)

    return [theta, errorHistory] 


def normalizeFeatures(X):
    X_norm = X
    mean = np.mean(X_norm, axis=0)
    sigma = np.std(X_norm, axis=0)
    X_norm = (X_norm-mean)
    X_norm = X_norm/sigma

    return [X_norm, mean, sigma]


# x array - > [ square feet, number of bedrooms ]
def predictPrice(arrX, theta):
    predict = np.dot(arrX, theta)
    return predict[0]


# numpy config. no scientific notation
np.set_printoptions(suppress=True)
# init vars
data = np.genfromtxt("ex1data2.txt", delimiter=',')
iterations = 400
alpha = 0.01

# initialize with all columns except last
X = np.array(data[:,:-1])
# initialize with last column from file
y = np.array([data[:,-1]])
y = np.transpose(y)

# adding a column of ones to X
ones = np.ones((X.shape[0], 1))
X = np.hstack((ones, X))

# passing without first column ( column of ones )
X[:,1:], mean, std = normalizeFeatures(X[:,1:])


theta = np.zeros((X.shape[1],1))
[theta, errorHistory] = gradientDescent(X, y, alpha, theta, iterations)

# PREDICTION
X1 = [1, 1650, 3]
X1[1:] = (X1[1:]-mean)/std
print(predictPrice(X1, theta))

# PLOTTING
# 2D
arrayIterations = np.transpose(np.arange(1, iterations+1))
plt.plot(arrayIterations, errorHistory, "r")
plt.show()
