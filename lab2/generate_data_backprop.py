import numpy as np
from scipy import signal

def functions_to_approx(train_datapoints, test_datapoints):
        training_set_sin=np.sin(2*train_datapoints)+np.random.normal(0,0.1,len(train_datapoints))
        test_set_sin=np.sin(2*test_datapoints)+np.random.normal(0,0.1,len(test_datapoints))

        training_set_square=signal.square(2*train_datapoints)+np.random.normal(0,0.1,len(train_datapoints))
        test_set_square=signal.square(2*test_datapoints)+np.random.normal(0,0.1,len(test_datapoints))

        return training_set_sin, test_set_sin, training_set_square, test_set_square


if __name__ == "__main__":
    training_points=np.arange(0, 2*np.pi, 0.1)
    test_points=np.arange(0.05, 2*np.pi, 0.1)
    
    training_set_sin_target, test_set_sin_target, training_set_square_target, test_set_square_target =functions_to_approx(training_points,test_points)
