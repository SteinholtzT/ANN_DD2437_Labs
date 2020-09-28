git pimport numpy as np

def createData():

    with open("/home/o/e/oerl/ANN/lab3/pict.dat", "r") as f:
        text = f.read().strip('\n').split(',')
        
        startIdx = 0
        endIdx = 1024
        data = np.empty((11, 1024))
        
        for i in range(11):
            temp = np.array(text[startIdx:endIdx])
            temp = np.reshape(temp, (1, 1024))
            data[i, :] = temp

            startIdx += 1024
            endIdx += 1024

    x1=[-1, -1, 1, -1, 1, -1, -1, 1]
    x2=[-1, -1, -1, -1, -1, 1, -1, -1]
    x3=[-1, 1, 1, -1, -1, 1, -1, 1,]

    X = np.empty((3, 8))
    X[0, :] = x1
    X[1, :] = x2
    X[2, :] = x3

    x1d=[1, -1, 1, -1, 1, -1, -1, 1]
    x2d=[1, 1, -1, -1, -1, 1, -1, -1]
    x3d=[1, -1, 1, -1, 1, 1, -1 ,-1]


    X_dist = np.empty((3, 8))
    X_dist[0, :] = x1d
    X_dist[1, :] = x2d
    X_dist[2, :] = x3d

    # print("data ", data[9:11, :])
    return X, X_dist, data[0:9, :], data[9:50, :]

if __name__ == "__main__":
    createData()    