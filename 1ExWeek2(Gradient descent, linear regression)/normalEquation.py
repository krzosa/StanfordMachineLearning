import numpy as np

# returns theta
def normalEquation(X, y):
    # adding a column of ones to X
    ones = np.ones((X.shape[0], 1))
    X = np.hstack((ones, X))

    transposeX = np.transpose(X) 
    # inverse(transposeX*X)*transposeX*y
    return np.linalg.pinv(transposeX.dot(X)).dot(transposeX.dot(y))


# input house -> [ houseInFeet^2, numberOfBedrooms ]
def predictHousePrice(house, theta):
    house = np.append(1, house)
    return house.dot(theta)



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

theta = normalEquation(X,y)
print(predictHousePrice(np.array([1650, 3]), theta))