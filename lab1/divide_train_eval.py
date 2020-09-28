
import numpy as np
import matplotlib.pyplot as plt


def divide_train_eval(classA,classB, percA,percB,n, findsample): #
        if findsample==False:
                shuffle = np.arange(n)
                np.random.shuffle(shuffle)
                subsetA=classA[:, shuffle[round(n-percA*n):n] ]
                subsetB=classB[:, shuffle[round(n-percB*n):n] ]

                classA=classA[:, shuffle[0:round(n-percA*n)] ]
                classB=classB[:, shuffle[0:round(n-percB*n)] ]

                subtargetA = np.ones((1, subsetA.shape[1]))
                subtargetA[:,:]=1

                subtargetB = np.zeros((1, subsetB.shape[1]))
                subtargetB[:,:]=-1

                Evaluate_set=np.concatenate([subsetA,subsetB],1)
                Evaluate_target=np.concatenate([subtargetA,subtargetB],1)

                size_patterns= classA.shape[1] + classB.shape[1]
        
                # Patterns (X)
                patterns = np.zeros((3, size_patterns))
                patterns[:, 0:round(n-percA*n)] = classA
                patterns[:, round(n-percA*n):size_patterns] = classB
        
                # Making targets (T)
                targets = np.zeros((1, size_patterns))
                targets[:, 0:round(n - percA*n)] = 1
                targets[:,round(n - percA*n): size_patterns] = -1

                return patterns, targets,  Evaluate_set, Evaluate_target, subsetA, subsetB

        if findsample == True:

                remove_lowA_idx = np.where(classA[1, :]<0)
                remove_lowA_idx=remove_lowA_idx[0][0:round(0.2*len(remove_lowA_idx[0])) ] #0.2 hardcoded for a 20% removal
                # valA_low = remove_lowA_idx[0][round(0.2*len(remove_lowA_idx[0])) : -1 ]

                remove_highA_idx=np.where(classA[1, :]>0)
                remove_highA_idx=remove_highA_idx[0][0:round(0.8*len(remove_highA_idx[0])) ] #0.8 hardcoded for a 80% removal, one scenario given
                # valA_high = remove_highA_idx[0][round(0.8*len(remove_highA_idx[0])) : -1 ]

                idx_vector=[remove_lowA_idx, remove_highA_idx]
                idx_vector=np.concatenate([remove_lowA_idx,remove_highA_idx])
                # idxVal = [valA_high, valA_low]
                # idxVal = np.concatenate([valA_high, valA_low] )

                # print(classA.shape)
                left_classA=np.delete(classA,idx_vector,1)
                np.delete(classA,idx_vector,1)
                # print(classA.shape)
                targets = np.zeros((1, n*2))
                targets[:,0: n] = 1
                targets[:,n:n*2] = -1

                patterns = np.zeros((3, n*2))
                patterns[:, 0:n] = classA
                patterns[:, n:n*2] = classB
                

                subset=patterns[:,idx_vector]
                targets_subset=targets[:, idx_vector]
                
                
                patterns= np.delete(patterns, idx_vector, 1)
                targets=np.delete(targets, idx_vector,1)
                
                plt.figure(99)
                plt.plot(patterns[0], patterns[1], 'ro')

                
                valB = np.zeros((subset.size))
                return patterns, targets, subset, targets_subset, subset, valB
