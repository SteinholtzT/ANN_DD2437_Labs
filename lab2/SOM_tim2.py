import numpy as np
from math import ceil
import matplotlib.pyplot as plt



def data():
    with open('./data/cities.dat', 'r') as f:
        text = f.read().split('\n')[4:]
        citiesCords = np.empty((10, 2))

        for i in range(len(text) - 1):
            citiesCords[i, :] = text[i].strip(';').split(', ')

    return citiesCords



class SOM:
    def __init__(self, attributes, gridX, gridY, epochs, eta):
        self.epochs = epochs
        self.attributes = attributes
        self.weights = np.random.uniform(size=(gridX,gridY))
        self.hood = 2
        self.eta = eta
        

    def train(self):
        for i in range(self.epochs):
            idx = np.arange(self.attributes.shape[0])
            # np.random.shuffle(idx)

            for j in range(self.attributes.shape[0]): #self.attributes.shape[0]

                diff = (self.attributes[idx[j], :] - self.weights)
                winner = np.argmin( np.diag( (diff) @ np.transpose(diff) ) )
                # print(" diff = ", np.diag((diff) @ np.transpose(diff))[winner] )
                hood_low = winner - self.hood
                hood_high = winner + self.hood
                if hood_low < 0:
                    hood_low = self.weights.shape[0] - hood_low
                    self.weights[0 : hood_high + 1] += self.eta * diff[0 : hood_high + 1]
                    self.weights[hood_low : -1] += self.eta * diff[hood_low : -1]
                elif hood_high > self.weights.shape[0]:
                    self.weights[0 : hood_high + 1] += self.eta * diff[0 : hood_high + 1]
                    self.weights[hood_low : -1] += self.eta * diff[hood_low : -1]
                else:
                    self.weights[hood_low : hood_high + 1, :] +=  self.eta * diff[hood_low : hood_high + 1, :]

            # self.eta = 0.95 * self.eta
            if i == 5:
                self.hood = 1
            if i == 10:
                self.hood = 0

        return self.weights       


    def output(self):
        winner = []
        for j in range(self.attributes.shape[0]): #self.attributes.shape[0]
            diff = (self.attributes[j, :] - self.weights)
            winner.append(np.argmin( np.diag( (diff) @ np.transpose(diff) ) ))
        return np.array(winner), self.weights

    def bestNode(self):
        bestWeights = np.empty((10,2))
        for j in range(self.attributes.shape[0]): #self.attributes.shape[0]
            diff = (self.attributes[j, :] - self.weights)

            idx = np.argmin( np.diag( (diff) @ np.transpose(diff) ) )
            print(np.argmin( np.diag( (diff) @ np.transpose(diff) ) ))
            bestWeights[j,:] = self.weights[idx, :]
            # bestWeights.append(np.argmin( np.diag( (diff) @ np.transpose(diff) ) ))

        self.weights = bestWeights
        # return self.weights
                
###################################################################################################################################################

citiesCords = data()

gridX = 25
gridY = 2
epochs = 20
eta = 0.2

som = SOM(citiesCords, gridX, gridY, epochs, eta)
weights = som.train()
som.bestNode()
output, weights1 = som.output()
idx = np.argsort(output)
# idx = sorted(output)
print(output)
print("sorted = ", idx)
print(" sorted out = ", output[idx])
# print("weights sorted = ", weights[idx])

path = citiesCords[idx]
# path1 = weights1[output]

plt.figure(1)
plt.scatter(citiesCords[:, 0], citiesCords[:, 1])
plt.scatter(weights[:,0], weights[:,1], s = 80, facecolors='none', edgecolors='r')
plt.scatter(weights1[:,0], weights1[:,1], s = 200, facecolors='none', edgecolors='y')
plt.plot(path[:, 0], path[:, 1], 'g--')
# plt.plot(path1[:, 0], path1[:, 1], 'k--')
plt.legend(( 'Citie Path', 'Cites', 'Nodes', 'Best Nodes'))

plt.figure(1)
for i in range(10):
    plt.annotate(str(i), xy=(citiesCords[i,0], citiesCords[i,1]))
    # plt.annotate(str(i), xy=(weights[i,0], weights[i,1]), color='blue')

plt.show()
