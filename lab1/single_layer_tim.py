
import numpy as np
import matplotlib.pyplot as plt
from shuffle import Shuffle

## Global variable
x = np.arange(-1,1, 0.1)


def deltaRule(X, T, W, eta, epochs, batchSz):
    itt = round(T.size / batchSz)
    error = []
    for _ in range(epochs):
        strtIdx = 0
        endIdx = batchSz
        for _ in range(itt):
            dw = -eta * np.matmul( (np.matmul(W, X[:, strtIdx:endIdx])  - T[:, strtIdx:endIdx]),  np.transpose(X[:, strtIdx:endIdx]))
            strtIdx += batchSz
            endIdx += batchSz
            W += dw

        prediction = W @ X
        prediction = np.where(prediction < 0, -1, 1)
        missClassified = np.where( np.subtract(T, prediction) != 0)[1]
        error.append(missClassified.size)

    bound = - (W[0,0] / W[0, 1]) * x - W[0, 2]/W[0, 1]
    return bound, error



def perceptron(X, T, W, eta, epochs, batchSz):
    itt = round(T.size / batchSz)
    bound = - (W[0,0] / W[0, 1]) * x - W[0, 2]/W[0, 1]
    for _ in range(epochs):
        activation = W @ X
        prediction = np.where(activation < 0, -1, 1)
        missClassified = np.where( np.subtract(T, prediction)!= 0 )[0]
        if(missClassified.size == 0):
            print("CONVERGED!")
            return bound

        bound = - (W[0,0] / W[0, 1]) * x - W[0, 2]/W[0, 1]
        W = W + eta * np.matmul( np.subtract(T, prediction), np.transpose(X) )

    return bound


# Values for defining classes
n = 100
mA = [1.0, 0.5]
sigmaA = 0.4
mB = [-1.0, -0.0]
sigmaB = 0.4

# Creating classes
classA = np.empty((2, n))
classB = np.empty((2, n))

# np.random.seed(10)
classA[0, :] = np.random.normal(mA[0], sigmaA, n)
classA[1, :] = np.random.normal(mA[1], sigmaA, n)
classB[0, :] = np.random.normal(mB[0], sigmaB, n)
classB[1, :] = np.random.normal(mB[1], sigmaB, n)

plt.figure(1)
plt.plot(classA[0], classA[1], 'bo', classB[0], classB[1], 'ro' )

# Creating the data
T = np.empty((1, 2*n))
T[0, 0:n] = 1
T[0, n:2*n] = -1
X = np.empty((3, 200))
X[0:2, 0:n] = classA
X[0:2, n:2*n] = classB
X[2, :] = 1
W = np.random.normal(0, 1, (1, 3))

# Shuffle the data
X, T = Shuffle(X, T)


# Call delta rule
eta = [0.001, 0.005, 0.01]
epochs = 100
batchSz = 200

epochsLen = np.arange(epochs)

# ####Delta rule
for i in range(len(eta)):
    bound, error = deltaRule(X, T, W, eta[i], epochs, 1)
    plt.figure(1) 
    plt.plot(x, bound)
    plt.figure(2)
    plt.plot(epochsLen, error)

plt.figure(1)
plt.legend(('classA', 'classB' , 'eta = 0.01' , 'eta = 0.05', 'eta = 0.1'))
plt.figure(2)
plt.legend(('eta = 0.001', 'eta = 0.005' , 'eta = 0.01' , 'eta = 0.05', 'eta = 0.1'))

# Sequntial with batchSz = 1
# bound, error = deltaRule(X, T, W, eta, epochs, 1)
# plt.figure(1)
# plt.plot(x, bound, 'g--')
# plt.figure(2)
# plt.plot(epochsLen, error, 'g')


### Perceptron
# bound = perceptron(X, T, W, eta, epochs, batchSz)
# plt.figure(1)
# plt.plot(x, bound, 'g--')


# Show plots
plt.show()

