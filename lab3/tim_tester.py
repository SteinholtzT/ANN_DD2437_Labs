import numpy as np
from createData import createData
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from random import randrange
from Hopfield_tim import HopfClass
from functions import generate_noise


############################## ENERGY
## Get data:
# X, X_dist, data, data_dist = createData()


# ## Set which patterns and inputs you want to have
# patterns = np.random.normal(0, 1, size=(1, 1024)) #np.copy(data[0:3, :])
# # patterns = np.random.normal(0, 1, size=(1, 1024)) #np.copy(data[0:3, :])
# # inputs = generate_noise(np.copy(inputs), 50 )
# inputs  = np.random.choice([-1,1], size=1024)

# # Initialize Hopfield Class 
# hopf = HopfClass(inputs, patterns, asyncUpdate=True, random=True, symetric=False, draw = False)


# energy = hopf.Energy(patterns)
# notConv, outPatterns, energyItt = hopf.findPattern()
# hopf.plotOut()
# # print(outPatterns)

# itterV = np.arange(energyItt.shape[0])
# plt.figure(1)
# plt.plot(itterV, energyItt[:, 0])
# # plt.plot(itterV, energyItt[:, 1])
# plt.axhline(energy[0], linestyle='--', color='red')
# # plt.axhline(energy[1], linestyle='--', color='green')
# # plt.axhline(energy[2], linestyle='--', color='black')
# plt.ylabel('Energy')
# plt.xlabel('Itteration')
# plt.legend(('Energy over time', 'Starting Energy'))






#############################################################    CAPACITY

#X, X_dist, data, data_dist = createData()

nrOfIn = [ 135]
# noiseV = [1, 10, 25, 35, 50, 60, 75]
noiseV = [0, 25, 50, 75, 100]

data = np.random.choice([-1,1], size=[135, 1024])

for j in nrOfIn:
    patterns = np.copy(data[0:j, :])
    inputs = np.copy(data[0:j, :])
    #print(j)

    errorV = np.zeros( (j, len(noiseV)) )

    for n in range(len(noiseV)):         
        #print(noiseV[n])   
        noisyIn = generate_noise(np.copy(inputs), noiseV[n])
        hopf = HopfClass(noisyIn, patterns, asyncUpdate=False, random=False, symetric=False, draw = False)
        
        _, outPatterns, _ = hopf.findPattern()
        
        diff = np.copy(patterns) - np.copy(outPatterns)
        for o in range(patterns.shape[0]):
            error = np.where(diff[o, :] !=0 )[0].size / patterns.shape[1]
            # print(" error = ", np.where(diff[o, :] !=0 )[0].size)
            #print(error)
            errorV[o, n] +=  (1 - error)            
    
    dataset = []
    plt.figure(j)
    for o in range(j):
        print("o = ", o+1)
        dataset.append(o+1)
        # plt.plot(noiseV, errorV[o, :], label = "p{}".format(dataset[o]))
        plt.plot(noiseV, errorV[o, :])
        plt.xlabel("Percentage of disorted patterns")
        plt.ylabel("Fraction of correctly classified data points")
        #plt.legend()

plt.show()
    




