import numpy as np
from math import ceil


def data():
    animals = open('./data/animalnames.txt', 'r')
    animals = np.array(animals.read().split())

    with open('./data/animals.dat', 'r') as f:
        text = f.read().split(',')
        text = list(map(int, text))
        attributes = np.empty((32, 84))

        start_idx = 0
        end_idx = 84
        for i in range(32):
            attributes[i, :] = text[start_idx : end_idx]
            start_idx += 84
            end_idx += 84
    return attributes, animals



class SOM:
    def __init__(self, attributes, gridX, gridY, epochs, eta):
        self.epochs = epochs
        self.attributes = attributes
        self.weights = np.random.uniform(size=(gridX,gridY))
        self.hood = 50
        self.eta = eta

    def train(self):
        hood_step = ceil(self.hood / self.epochs)
        for _ in range(self.epochs):
            for j in range(self.attributes.shape[0]): #self.attributes.shape[0]

                diff = (self.attributes[j, :] - self.weights)
                winner = np.argmin( np.diag( (diff) @ np.transpose(diff) ) )
                hood_low = winner - self.hood
                hood_high = winner + self.hood
                if hood_low < 0:
                    hood_low = 0
                if hood_high > self.weights.shape[0]:
                    hood_high = self.weights.shape[0]
                self.weights[hood_low : hood_high + 1, :] +=  self.eta * diff[hood_low : hood_high + 1, :]
            
            self.hood -= hood_step
            if(self.hood < 0):
                self.hood = 0

    def output(self):
        winner = []
        for j in range(self.attributes.shape[0]): #self.attributes.shape[0]
            diff = (self.attributes[j, :] - self.weights)
            winner.append(np.argmin( np.diag( (diff) @ np.transpose(diff) ) ))
        return np.array(winner)
                
###################################################################################################################################################

attributes, animals = data()

gridX = 100
gridY = 84
epochs = 20
eta = 0.2

som = SOM(attributes, gridX, gridY, epochs, eta)
som.train()
output = som.output()
idx = np.argsort(output)

print(output[idx])
print(animals[idx])
