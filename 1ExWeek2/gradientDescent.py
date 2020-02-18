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
def gradientDescent(X, y, alpha, numberOfIterations):
    m = y.size
    errorHistory = np.zeros((numberOfIterations, 1))
    theta = 0

    for i in range(0, numberOfIterations):
        # theta = theta - (alpha/m)*X'*(X*theta - y)
        # where:
        # X' - transpose of X
        theta = theta - (alpha/m)*np.dot(np.transpose(X), (np.dot(X,theta) - y))
        errorHistory[i] = computeCost(X, y, theta)

    return [theta[:,0], errorHistory] 


def predictProfit(populationTensOfThousands, theta):
    predict = np.dot(np.array([populationTensOfThousands, 1]), theta)
    return predict * 10000


def main():
    # init variables
    data = np.genfromtxt("ex1data1.txt", delimiter=',')
    theta = np.zeros((2,1))
    iterations = 1500
    alpha = 0.01

    # X, y
    population = np.array([data[:,0]])
    profit = np.array([data[:,1]])
    X = np.transpose(population)
    y = np.transpose(profit)

    # adding a column of ones to X
    ones = np.ones((X.shape[0], 2))
    ones[:,0] = X[:,0]
    X = ones

    # calculating theta and errorHistory with gradientDescent algorithm
    [theta, errorHistory] = gradientDescent(X, y, alpha, iterations)

    # PREDICTION

    # predicting profit for population of 35000 and 70000
    print("Predicted profit when 35000")
    print(predictProfit(3.5, theta))
    print("Predicted profit when 70000")
    print(predictProfit(7, theta))

    # PLOTTING
    # 2D
    # iterations x error 
    plt.subplot(2,1,1)
    arrayIterations = np.transpose(np.arange(1, iterations+1))
    plt.plot(arrayIterations, errorHistory, "r")

    # # population x profit
    plt.subplot(2,1,2)
    plt.xlabel("population")
    plt.ylabel("profit")
    plt.plot(X[:,0], y, "rx", X[:,0], np.dot(X,theta), "b")
    plt.show()


if __name__ == "__main__":
    main()