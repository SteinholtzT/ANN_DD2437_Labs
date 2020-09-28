import numpy as np
import matplotlib.pyplot as plt
from math import exp
from shuffle import Shuffle
from divide_train_eval import divide_train_eval

    
class  GenDelta:
    def __init__(self, patterns, targets, valPatterns, valTargets, nrOfHiddenN, nrOfOutN, epochs, alpha, eta, batchSz):
        self.valPatterns = valPatterns
        self.valTargets = valTargets
        self.batchSz = int(batchSz)
        self.alpha = alpha
        self.eta = eta
        self.nrOfHiddenN = nrOfHiddenN
        self.epochs = epochs
        self.patterns = patterns
        self.targets = targets
        self.hiddenLayer = np.empty((nrOfHiddenN, patterns.shape[0]))
        self.outputLayer = np.empty((nrOfOutN,  (nrOfHiddenN + 1) ))


        for i in range(nrOfHiddenN):
            self.hiddenLayer[i, :] = np.random.randn(1, self.hiddenLayer.shape[1])
        for i in range(nrOfOutN):
            self.outputLayer[i, :] = np.random.randn(1, self.outputLayer.shape[1])
            

        self.dw = np.zeros( (self.hiddenLayer[0].shape) )
        self.dv = np.zeros( (self.outputLayer[0].shape) )

    def train(self):
        missClassifiedVec = []
        MeanSQVec = []
        missClassifiedVecVal = []

        itterations = self.patterns.shape[1]//self.batchSz
        
        for _ in range (self.epochs):
            strtIdx = 0
            endIdx = self.batchSz
            #Sequential
            for _ in range(itterations):
                    # print("IIIIIIIII ",  i)
                hOut, oOut = self.forward(self.patterns[:, strtIdx:endIdx])
                deltaO, deltaH = self.backward(self.outputLayer, self.targets[:, strtIdx:endIdx], oOut, hOut, self.nrOfHiddenN)
                self.weightUpdate(self.patterns[:, strtIdx:endIdx], deltaO, deltaH, hOut)
                strtIdx += self.batchSz
                endIdx += self.batchSz

                hOut, oOut = self.forward(self.patterns)
                _, valOut = self.forward(self.valPatterns)

            missClassError, MeanSQError, indexes = self.calcError(oOut, self.targets)
            missClassifiedVec.append(missClassError)
            MeanSQVec.append(MeanSQError)
            missClassErrorVal, _, indexesVal = self.calcError(valOut, self.valTargets)
            missClassifiedVecVal.append(missClassErrorVal)

        # indexesVal = 1
        return missClassifiedVec, MeanSQVec, indexes, missClassifiedVecVal, indexesVal


    def forward(self, inputs): 
        hin = self.hiddenLayer @ inputs
        hout = ( 2 / (1+np.exp(-hin)) ) - 1
        hout = np.vstack((hout, np.ones(hout.shape[1])))
        oin = self.outputLayer @ hout
        oout = ( 2 / (1+np.exp(-oin)) ) - 1

        return hout, oout

    def backward(self, v, targets, oOut, hOut, nrOfHiddeN):
        deltaO = (oOut - targets) * ((1+oOut) * (1-oOut))*0.5
        deltaH = ( np.transpose(v) @ deltaO) * ((1+hOut) * (1-hOut))*0.5
        deltaH = deltaH[0:-1, :]
        return deltaO, deltaH


    def weightUpdate(self, patterns, deltaO, deltaH, hOut):
        self.dw = ( self.dw * self.alpha ) - (deltaH @ np.transpose(patterns)) * (1-self.alpha)
        self.dv = ( self.dv * self.alpha ) - (deltaO @ np.transpose(hOut)) * (1-self.alpha)
        self.hiddenLayer = self.hiddenLayer + self.dw * self.eta
        self.outputLayer = self.outputLayer + self.dv * self.eta
        

    def calcError(self, Out, targets):
        # _ , Out = self.forward(self.patterns)
        rootMean =  np.mean (np.square( Out- targets ) )
        Out = np.where(Out < 0, -1, 1)
        missClassified = Out - targets
        indexes = np.where(missClassified != 0)[1]
        missClassified = np.where(missClassified != 0)[1].size / targets.size
        

        return missClassified, rootMean, indexes
     



def standardDev(inputList):
    # print()
    # print("inputlist, ", inputList)
    mean = sum(inputList) / len(inputList) 
    variance = sum([((x -  mean) ** 2) for x in inputList]) / len(inputList) 
    res = variance ** 0.5
    return res



def generate_classes(n, sigmaA, mA, sigmaB, mB):

    nHl = round(n / 2)

    classA = np.ones((3, n))
    classA[0, 0:nHl] = np.random.randn(1, nHl) * sigmaA - mA[0]
    classA[0, nHl:n] = np.random.randn(1, nHl) * sigmaA + mA[0]
    classA[1, 0:n] = np.random.randn(1, n) * sigmaA + mA[1]  

    classB = np.ones((3, n))
    classB[0, 0:n] = np.random.randn(1, n) * sigmaB + mB[0]
    classB[1, 0:n] = np.random.randn(1, n) * sigmaB + mB[1]

    targets = np.zeros((1, n*2))
    targets[0, 0:n] = 1
    targets[0, n:n*2] = -1
    
    patterns = np.zeros((3, n*2))
    patterns[:, 0:n] = classA
    patterns[:, n:n*2] = classB
    
    return patterns, targets, classA, classB
##########################################################



n = 500
mA = [1.0, 0.3]
sigmaA = 0.2
mB = [0.0, -0.1]
sigmaB = 0.3


nrOfNeuronsH = [25]

nrOfNeuronsO = 1
epochs = 2000
alpha = 0     # alpha = 0 for no momentum
eta = 0.001
batchSz = n / 5

nrOfAvarage = 1

MissClassed = np.empty((len(nrOfNeuronsH), nrOfAvarage, epochs  ))
valMiss = np.empty((len(nrOfNeuronsH), nrOfAvarage, epochs  ))
MSE = np.empty((len(nrOfNeuronsH), nrOfAvarage, epochs  ))
meanMissCl = np.empty((len(nrOfNeuronsH), epochs))
meanVal = np.empty((len(nrOfNeuronsH), epochs))
meanMSE = np.empty((len(nrOfNeuronsH), epochs))


########### standard Deviation
# /(n, sigmaA, mA, sigmaB, mB):
for i in range(len(nrOfNeuronsH)):
    for j in range(nrOfAvarage):

        X, T, classA, classB = generate_classes(n, sigmaA, mA, sigmaB, mB)
        X, T, valX, valT, subclassA, subclassB = divide_train_eval(classA,classB, 0.25, 0.25 ,n, False)

        neuronLayer = GenDelta(X, T, valX, valT, nrOfNeuronsH[i], nrOfNeuronsO, epochs, alpha, eta, batchSz)
        missClassifiedError, MeanSQError, indexes, errorVal, indexesVal = neuronLayer.train()
        MissClassed[i, j, :] = missClassifiedError
        MissClassed[i, j , :] = MissClassed[i,j,:]*100
        valMiss[i, j, :] = errorVal
        valMiss[i, j , :] = valMiss[i,j,:]*100
        MSE[i, j, :] = MeanSQError

    meanMissCl[i, :] = np.sum(MissClassed[i, :, :], axis=0 ) / nrOfAvarage
    meanVal[i, :] = np.sum(valMiss[i, :, :], axis=0 ) / nrOfAvarage
    meanMSE[i, :] = np.sum(MSE[i, :, : ], axis=0) /nrOfAvarage


# stdDevMissCl = np.zeros( (len(nrOfNeuronsH), epochs) )
# stdDevVal =  np.zeros( (len(nrOfNeuronsH), epochs) )
# stdDevMSE = np.zeros( (len(nrOfNeuronsH), epochs) )
# print("MIss = ", MissClassed.shape)
# start = 20 
# for i in range(len(nrOfNeuronsH)):
#     start += 2
#     for j in range(start, epochs, 25):
#         stdDevMSE[i, j] = standardDev(MSE[i, :, j] )
#         stdDevMissCl[i, j] = standardDev(MissClassed[i, :, j] )
#         stdDevVal[i, j] = standardDev(valMiss[i, :, j] )

# print("Miss class err = ", meanMissCl[ 0, (epochs-1)])
# print("stdDev Miss = ", stdDevMSE[0, j])
# print("val err = ", meanVal[0, (epochs-1)])
# print("stdDev Val = ", stdDevVal[0, j])





###################### PLOT
classA = subclassA
classB = subclassB
indexes = indexesVal
X = valX


resolution = 100
x_v = np.linspace(-2, 2, resolution).reshape(1,resolution)
y_v = np.linspace(-2, 2, resolution).reshape(1,resolution)

# x_vec = x_val
x = x_v
for i in range(resolution-1):
    x = np.hstack((x,x_v))
grid = np.vstack((x, x))

index = 0
for y in y_v[0]:
    for i in range(resolution):
        grid[1,index] = y
        index += 1

grid = np.vstack((grid, np.ones(grid.shape[1])))

_, gridOut= neuronLayer.forward(grid)
z = gridOut.reshape(resolution,resolution)
x = grid[0].reshape(resolution,resolution)
y = grid[1].reshape(resolution,resolution)


RightClassified = np.delete(X[0,:], indexes)
RightClassified = np.vstack(( RightClassified, np.delete(X[1,:], indexes)))

WrongClassied = X[0,indexes]
WrongClassied = np.vstack((WrongClassied, X[1,indexes]))

plt.figure(3)
plt.contourf(x, y, z, cmap ="bone")
plt.colorbar()
plt.scatter(RightClassified[0,:], RightClassified[1,:], s=80, c='g')
plt.scatter(WrongClassied[0,:], WrongClassied[1,:], s=50, c='r')
plt.scatter(classA[0], classA[1], s=15, marker='+',c="k")
plt.scatter(classB[0], classB[1], s= 15, marker='^',c="k") 
plt.ylabel('y')
plt.xlabel('x')
plt.title('Decision-boundary, Nr of Neurons = 25')


# epochV = np.arange(epochs)
# plt.figure(1)
# plt.errorbar(epochV, meanMissCl[0,:], stdDevMissCl[0,:])
# # plt.errorbar(epochV, meanMissCl[1,:], stdDevMissCl[1,:])
# # plt.errorbar(epochV, meanMissCl[2,:], stdDevMissCl[2,:])
# plt.ylabel('Accuracy [%] ')
# plt.xlabel('Epoch')
# plt.title('Classification Accuracy')
# plt.legend(('Nr of Neurons = 1','Nr of Neurons = 10', 'Nr of Neurons = 25' ))

# plt.figure(2)
# plt.errorbar(epochV, meanVal[0,:], stdDevVal[0,:])
# # plt.errorbar(epochV, meanVal[1,:], stdDevVal[1,:])
# # plt.errorbar(epochV, meanVal[2,:], stdDevVal[2,:])
# plt.ylabel('Accu')
# plt.xlabel('Epoch')
# plt.title('Validation')
# plt.legend(('Nr of Neurons = 1','Nr of Neurons = 10', 'Nr of Neurons = 25' ))



# plt.figure(1)
# plt.plot(epochV, MissClassed[0,0,:], 'r')
# plt.ylabel('Percent of miss classified')
# plt.xlabel('Number of nodes')
# plt.title('Percent of miss classified after 1000 epochs')

# plt.figure(2)
# plt.plot(epochV, missClassifiedError, 'r')
# plt.ylabel('Percent of miss classified')
# plt.xlabel('Number of nodes')
# plt.title('Percent of miss classified after 1000 epochs')

# plt.figure(2)
# plt.plot(nrOfNeuronsH, lastMSE, 'b')
# plt.ylabel('MSE')
# plt.xlabel('Number of nodes')
# plt.title('MSE after 1000 epochs')


plt.show()










