import numpy as np

def Shuffle(In, Out):
    length = Out.shape[1]
    shuffle = np.arange(length)
    np.random.shuffle(shuffle)

    In = In[:, shuffle]
    Out = Out[:, shuffle]

    return In, Out