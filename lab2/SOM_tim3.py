import numpy as np
from math import ceil
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec


def data():
    with open('./data/mpparty.dat', 'r') as f:
        text = f.read().split()[20:]
        party = text
        # print(text)
    
    with open('./data/mpsex.dat', 'r') as f:
        text = f.read().split()[6:]
        gender = text
        # print(gender)

    with open('./data/mpdistrict.dat', 'r') as f:
        text = f.read().split()
        district = text
        print(district)

    with open('./data/votes.dat', 'r') as f:
        text = f.read().split(',')
        votes = np.empty((349, 31))

        strtIdx = 0
        endIdx = 31
        for i in range(349):
            votes[i, :] = text[strtIdx : endIdx]
            strtIdx += 31
            endIdx += 31

    return np.array(votes), np.array(party), np.array(gender), np.array(district)


class SOM:
    def __init__(self, attributes, gridX, gridY, epochs, eta):
        self.epochs = epochs
        self.attributes = attributes
        self.weights = np.random.uniform(size=(gridX, gridY, 31))
        self.hood = 2
        self.eta = eta
        self.gridX = gridX
        self.gridY = gridY

        # self.weights = np.reshape(self.weights, (100, 31))

    def train(self):
        for i in range(self.epochs):
            for j in range(self.attributes.shape[0]): 


                diff = (self.attributes[j, :] - self.weights)
                dist = np.linalg.norm(diff, axis=2)
                idx = np.where(dist == np.min(dist))
                idxX = idx[0][0]
                idxY = idx[1][0]
                lowX = idxX - self.hood
                highX = idxX + self.hood
                lowY = idxY - self.hood
                highY = idxY + self.hood


                if lowX < 0:
                    lowX = 0
                if lowY < 0:
                    lowY = 0
                if highX > gridX:
                    highX = gridX
                if highY > gridY:
                    highY = gridY

          
                self.weights[lowX:highX + 1, lowY:highY+1, :] +=  self.eta * diff[lowX:highX + 1, lowY:highY+1, :]


            if i == 5:
                self.hood = 1
            if i == 10:
                self.hood = 0
            if i == 15:
                self.hood = 0 

    def output(self):
        pos = np.empty((self.attributes.shape[0], 1))
        # print(self.attributes.shape)

        winner = []


        for j in range(self.attributes.shape[0]): #self.attributes.shape[0]
            diff = (self.attributes[j, :] - self.weights)
            dist = np.linalg.norm(diff, axis=2)
            idx = np.where(dist == np.min(dist))
            idxX = idx[0][0]
            idxY = idx[1][0]
            winner.append( [idxX, idxY] )

        return np.array(winner)
                


class SHOW:
    def __init__(self):
        self.grid = {}
        for x in range(10):
            for y in range(10):
                self.grid[x, y] = []

    def update(self, pos, input):
        self.grid[pos[0], pos[1]].append(input)

    def unique(self, list1): 
        unique_list = [] 
        for x in list1: 
            if x not in unique_list: 
                unique_list.append(x) 
        return unique_list
      
    def plot(self):
        colors = ['k', '#0000ff', '#3399ff', '#ff0000', '#990000', '#339933', '#6600cc', '#00ffcc'] # For parties
        # colors = ['b', 'r'] # For genders

        fig, ax = plt.subplots()
        for x in range(10):
            for y in range(10):
                voters = self.grid[x, y]
                print("voters = ", voters)
                if len(voters) >0:
                    uniq = self.unique(voters)
                    dominant = []
                    for i in range(len(uniq)):
                        dominant.append(voters.count(uniq[i]) )

                    dom = uniq[dominant.index(max(dominant))]
                    perc = max(dominant) / len(voters)

                    circle1 = plt.Circle((x, y), 0.5, color=colors[int(dom)])
                    ax.annotate(str(round(float(perc)*100)) + "%", xy=(x, y), fontsize=7, ha="center")
                    ax.add_artist(circle1)
                    plt.ylim(-1, 10)
                    plt.xlim(-1, 10)
        plt.show()


    def plot2(self):
        # PartyColors = ['k', '#0000ff', '#3399ff', '#ff0000', '#c40b0b', '#339933', '#6600cc', '#49ff33'] # For parties
        GenderColors = ['b', 'r']
        DistrictColors = [
        "#FF0000",
        "#FF0049",
        "#FF008B",
        "#FF00F7",
        "#E400FF",
        "#AA00FF",
        "#6C00FF",
        "#2E00FF",
        "#3E39A0",
        "#7D79D8",
         "#79D3D8",
         "#33E7F2",
         "#33F2B5",
         "#1F7A3F",
         "#164626",
         "#111D15",
         "#CFCF1C",
         "#A0A019",
         "#4F4F16",
         "#E4B633",
         "#DF821F",
         "#874C0E",
         "#7C0E87",
         "#DE95E5",
         "#BDBDBD",
         "#3C6F8A",
         "#BB8ED5",
         "#4E2F61",
         "#7FEEFF"
        ]


        plt.figure(1)
        the_grid = GridSpec(10, 13)
        for x in range(10):
            for y in range(10):
                voters = self.grid[x, y]
                # print("voters = ", voters)
                if len(voters) >0:
                    uniq = self.unique(voters)
                    dominant = []
                    colors = []
                    for i in range(len(uniq)):
                        dominant.append(voters.count(uniq[i]) )
                        colors.append(DistrictColors[int(uniq[i])-1])
                    dominant = np.array(dominant)
            
                    print(uniq)
                    fracs = dominant / len(voters)
                    votes = (len(voters) / 349)
                    # voteFracs = [votes, (1-votes)]
              

                    plt.subplot(the_grid[x,y], aspect = 1)
                    plt.pie(fracs, colors=colors, radius=1.5, wedgeprops=dict(width=1.3, edgecolor='w')  )
            
        plt.subplot(the_grid[0, 12], aspect = 1)
        plt.pie([0, 0, 0, 0,0,0,0, 0,0, 0, 0, 0,0,0,0, 0, 0, 0, 0, 0,0,0,0, 0,0, 0, 0, 0,0,0,0, 0], colors=DistrictColors, radius=0)
        # plt.legend(('No party', 'm', 'fp', 's', 'v', 'mp', 'kd', 'c'))
        # plt.legend(('Male', ' Female'))
        plt.legend(('1', '2', '3', '4', '5', '6', '7', '8', '9', '10' ,'11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28'))
        # % Coding: 0=no party, 1='m', 2='fp', 3='s', 4='v', 5='mp', 6='kd', 7='c'
        plt.show()
    
    
    def printFun(self):
        print(self.grid)

    

###################################################################################################################################################

votes, party, gender, district = data()

# print("party =", party)
gridX = 10
gridY = 10
epochs = 20
eta = 0.2

som = SOM(votes, gridX, gridY, epochs, eta)
som.train()
pos = som.output()
# idx = np.argsort(output)



# % Coding: 0=no party, 1='m', 2='fp', 3='s', 4='v', 5='mp', 6='kd', 7='c'
colors = ['k', '#0000ff', '#3399ff', '#ff0000', '#990000', '#339933', '#6600cc', '#00ffcc']

show = SHOW()

for i in range(votes.shape[0]): #votes.shape[0]
    # party_nr = gender[i]
    show.update(pos[i], district[i])
    
show.plot2()
