import numpy as np
from createData import createData
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from random import randrange


class HopfClass:
    def __init__(self, inputs, outputs, random=False, symetric=False, asyncUpdate = False, draw = False, sparse = False, bias = 0):
        self.asyncUpdate = asyncUpdate
        self.draw = draw
        self.sparse = sparse
        self.bias = bias

        if (inputs.shape[0] == 1024 or inputs.shape[0] == 8) and sparse==False:
            self.inputs = np.reshape(inputs, (1, inputs.shape[0]))
        else:
            self.inputs = inputs
        
        if (outputs.shape[0] == 1024 or outputs.shape[0] == 8) and sparse==False:
            self.outputs = np.reshape(outputs, (1, outputs.shape[0]))
        else: self.outputs=outputs

        self.NotConv = np.zeros((1, self.inputs.shape[0]))


        if random:
            self.W = np.random.normal(0, 1, size=(outputs.shape[1], outputs.shape[1]))
            if symetric:
                self.W = 0.5 * (self.W + np.transpose(self.W)  )
            
            
        elif sparse:
            activity = np.where(inputs == 1)
            rho = activity[1].shape[0]/(inputs.shape[0]*inputs.shape[1])
            
            self.W = np.zeros((outputs.shape[1], outputs.shape[1]))
            x = np.empty((1, outputs.shape[1]))
            for p in range(outputs.shape[0]):
                x[0, :] = outputs[p, :]
                self.W += np.transpose(x - rho) * (x - rho)
            # self.W = self.W / outputs.shape[1]
            # print(self.W)
            
        else:
            self.W = np.zeros((outputs.shape[1], outputs.shape[1]))
            x = np.empty((1, outputs.shape[1]))
            for p in range(outputs.shape[0]):
                x[0, :] = outputs[p, :]
                self.W += np.transpose(x) * x
            self.W = self.W / outputs.shape[1]


    def update(self, i):
        if self.asyncUpdate:
            rand = randrange(0, self.W.shape[0])
            weights = np.reshape(self.W[rand, : ], (1, self.W.shape[0]))
            inp = np.reshape(self.inputs[i, :], (self.W.shape[0], 1) )
            new_value = weights @ inp

            if new_value < 0:
                self.inputs[i, rand] = -1
            if new_value > 0:
                self.inputs[i, rand] = 1
        elif self.sparse:
            rand = randrange(0, self.W.shape[0])
            weights = np.reshape(self.W[rand, : ], (1, self.W.shape[0]))
            inp = np.reshape(self.inputs[i, :], (self.W.shape[0], 1) )
            new_value = weights @ inp - self.bias
            
            if new_value < 0:
                self.inputs[i, rand] = 0
            if new_value > 0:
                self.inputs[i, rand] = 1
            
        else: 
            out = self.W @ np.transpose(self.inputs[i,:])
            self.inputs[i, :] = np.transpose(np.where(out > 0, 1., -1.))


    def findPattern(self):

       
        itters = 500
        # energyItt = np.array((self.inputs.shape[0], itters))
        energy = []
        plotItt = 100
        for itt in range(itters):
            # print("itt = ", itt)
            NotCon = np.where(self.NotConv == 0)[1]
            for i in range(NotCon.size):
                self.update(NotCon[i])
          
                energy.append(self.Energy(self.inputs))

                for o in range(self.outputs.shape[0]):
                    diff = np.where((self.inputs[NotCon[i],:] - self.outputs[o,:]) != 0 )[0].size
                    # print("Input: {}, Output: {}, MissCl: {} ".format(NotCon[i], o, diff))
                    if (diff == 0):
                        self.NotConv[0, NotCon[i]] = itt+1

            if itt == plotItt and self.draw:
                self.plotOut()
                plt.draw()
                plt.pause(0.0000001)
                plotItt += 100

            nr_not_conv = np.where(self.NotConv == 0)[1].shape
            if nr_not_conv[0] == 0:
                break
            
        return self.NotConv, self.inputs, np.array(energy)



    def Energy(self, X_in):
        energy=[]
        if X_in.shape[0] == 1024 or X_in.shape[0] == 8:
            X_in = np.reshape(X_in, (1, X_in.shape[0]))

        for i in range(X_in.shape[0]):
            energy.append(- X_in[i, :] @ self.W @ np.transpose(X_in[i, :]) )

        return energy


    def plotPatterns(self):
        if self.outputs.shape[1] != 1024:
            return

        plt.figure(1)
        for i in range(self.outputs.shape[0]):
            plt.subplot(1, self.outputs.shape[0], i+1)
            test = np.reshape(self.outputs[i,:], (32,32))
            plt.imshow(test)
    

    def plotOut(self):
        print(self.inputs.shape[1])        
        if self.inputs.shape[1] != 1024:
            return

        plt.figure(2)
        for i in range(self.inputs.shape[0]):
            plt.subplot(1, self.inputs.shape[0], i+1)
            test = np.reshape(self.inputs[i,:], (32,32))
            plt.imshow(test)

