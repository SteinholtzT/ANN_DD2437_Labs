import numpy as np

def generate_noise(X_in, procent_noise, sparse = False):
    patterns, datapoints = X_in.shape
    random_datapoints = round(datapoints*procent_noise/100)
    
    points_to_shuffle = np.sort(np.random.choice(datapoints, random_datapoints, replace=False))
    
    if sparse:
        X_in[:, points_to_shuffle] = np.where(X_in[:, points_to_shuffle] == 0, 1, 0)   
    else:    
        X_in[:, points_to_shuffle] = X_in[:, points_to_shuffle]*-1
        
    return X_in

def converged_check(In, Out):
    
    x, _ = np.where((In - Out) != 0 )
    x = np.unique(x)
    
    converged  = np.ones(In.shape[0])
    
    converged[x] = 0
    
    return converged