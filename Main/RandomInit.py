#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import essential modules
import numpy as np
import pickle
import sys

noTarget = 1 # target will always be one
gridHeightList = [int(sys.argv[1])]
gridWidthList = [int(sys.argv[1])]
noAgentList = [int(sys.argv[2])]
noObsList = [int(sys.argv[3])]
eList = [int(sys.argv[4])]
LoopVal = int(sys.argv[5]) # defines how many times the code will run
neighborWeightsList = [float(sys.argv[6])]



print("C"+str(len(gridWidthList))+ "_L"+str(LoopVal)+
              "_H"+str(gridHeightList[0])+"_W"+str(gridWidthList[0])+"_N"+str(noAgentList[0])+
              "_O"+str(noObsList[0])+"_E"+str(eList[0])+"_Nw"+str(neighborWeightsList[0]))


# initializing the lists
tPosList = []
aPosList = []
aPosListTotal = []
aPosListLoopTotal = []
aPosListCriteriaTotal = []

# finding out the target's position which is the last cell of the grid
for t in range(noTarget):
    tPosList.append([(int(gridHeightList[0]/(t+1))-1), (int(gridWidthList[0]/(t+1))-1)])

# main code
for CriteriaVal in range(len(gridWidthList)):
    height = gridWidthList[CriteriaVal]
    width = gridHeightList[CriteriaVal]
    Agent = noAgentList[CriteriaVal]
    Obs = noObsList[CriteriaVal]
    epoch = eList[CriteriaVal]
    neighborWeights = neighborWeightsList[CriteriaVal]

    
    for Loop in range(LoopVal):
        sX =[i for i in range(height-1)]
        sY =[i for i in range(width-1)]
        for ep in range(0,epoch):
            for a in range(Agent):
                aPosX = np.random.choice(sX)
                aPosY = np.random.choice(sY)
                if [aPosX, aPosY] not in tPosList:
                    aPosX= aPosX
                    aPosY= aPosY
                else:
                    aPosX = np.random.choice(sX)
                    aPosY = np.random.choice(sY)
                aPosList.append([aPosX, aPosY])
            aPosListTotal.append(aPosList)
            aPosList =[]
        aPosListLoopTotal.append(aPosListTotal)
        aPosListTotal=[]
        
    p = aPosListLoopTotal
    
    # saving the positions in a pickle file. C: number of different scenarios, L: number of times the code will run as a loop,
    # H: Grid Height, W: Grid Width, N: number of agents, O: number of obstacles, E: Total Episode, Nw: Neighbor weights
    with open("./RandomPosition/C"+str(CriteriaVal)+ "_L"+str(LoopVal)+
              "_H"+str(height)+"_W"+str(width)+"_N"+str(Agent)+
              "_O"+str(Obs)+"_E"+str(epoch)+"_Nw"+str(neighborWeights), "wb") as Pp:   #Pickling
        pickle.dump(p, Pp)
        
print("################# Random Initialization Done. #########################")
print("################# Please find the pickle file inside RandomPosition folder #########################")
