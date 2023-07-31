# importing essential modules
import numpy as np
import pickle
import random
import math
import sys

class Env:
    def __init__(self, gridHeight, gridWidth, playMode, noTarget, noAgent, noObs, noFreeway):
        self.height = gridHeight
        self.width = gridWidth
        self.courierNumber = 1

        # total states
        
        self.sX =[i for i in range(self.height-1)]
        self.sY =[i for i in range(self.width-1)]

        # initialize targets' position
        
        self.tPosList = []
        if playMode['Target'].lower() == 'random':
            for t in range(noTarget):
                tPosX = np.random.choice(self.sX)
                tPosY = np.random.choice(self.sY)
                self.tPosList.append([tPosX, tPosY])
        elif playMode['Target'].lower() == 'static':
            for t in range(noTarget):
                self.tPosList.append([(int(self.height/(t+1))-1), (int(self.width/(t+1))-1)])
        else:
            for t in range(noTarget):
                self.tPosList.append([self.height-1,self.width-1])
        

        # initialize agents' position
        
        self.aPosList = []
        if playMode['Agent'].lower() == 'random':
            for a in range(noAgent):
                aPosX = np.random.choice(self.sX)
                aPosY = np.random.choice(self.sY)
                if [aPosX, aPosY] not in self.tPosList:
                    self.aPosList.append([aPosX, aPosY])
                else:
                    aPosX = np.random.choice(self.sX)
                    aPosY = np.random.choice(self.sY)
                    self.aPosList.append([aPosX, aPosY])
        elif playMode['Agent'].lower() == 'static':
            for a in range(noAgent):
                self.aPosList.append([0,0])
        else:
            for a in range(noAgent):
                self.aPosList.append([0,0])


        # initialize obstacles' position
        
        self.oPosList = []
        if playMode['Obstacle'].lower() == 'random':
            for o in range(noObs):
                oPosX = np.random.choice(self.sX)
                oPosY = np.random.choice(self.sY)
                if (([oPosX, oPosY] not in self.tPosList) and ([oPosX, oPosY] not in self.aPosList)):
                    self.oPosList.append([oPosX, oPosY])
                else:
                    oPosX = np.random.choice(self.sX)
                    oPosY = np.random.choice(self.sY)
                    self.aPosList.append([oPosX, oPosY])
        elif playMode['Obstacle'].lower() == 'static':
            for o in range(noObs):
                self.oPosList.append([(int(self.height/(o+1))-1), (int(self.width/(o+1))-2)])
        else:
            for o in range(noObs):
                self.oPosList.append([0,0])


        # initialize freeways' position
        
        self.fPosList = []
        if playMode['Freeway'].lower() == 'random':
            for f in range(noFreeway):
                fPosX = np.random.choice(self.sX)
                fPosY = np.random.choice(self.sY)
                if (([fPosX, fPosY] not in self.tPosList) and ([fPosX, fPosY] not in self.aPosList) and ([fPosX, fPosY] not in self.oPosList)):
                    self.fPosList.append([fPosX, fPosY])
                else:
                    fPosX = np.random.choice(self.sX)
                    fPosY = np.random.choice(self.sY)
                    self.fPosList.append([fPosX, fPosY])
        elif playMode['Freeway'].lower() == 'static':
            for f in range(noFreeway):
                self.fPosList.append([(int(self.height/(f+1))-2), (int(self.width/(f+1))-1)])
        else:
            for f in range(noFreeway):
                self.fPosList.append([0,0])



        self.actions = [0, 1, 2, 3] # action list
        self.stateCount = self.height*self.width # counting states
        self.actionCount = len(self.actions) # total action count

    # reset function    
    def reset(self, playMode, noTarget, noAgent, noObs, noFreeway, gridHeight, gridWidth, epochVal, CriteriaVal, countVal, neighborWeights, totalEpisode,LoopVal):
        # initializing lists and variables
        self.aPosList = []
        self.tPosList = []
        self.oPosList = []
        self.fPosList = []
        self.doneList = []
        self.rewardList = []
        self.stateList = []
        self.courierNumber = 1

        # reset agents' position
        
        with open("RandomPosition/C"+str(0)+ "_L"+str(LoopVal)+
              "_H"+str(gridHeight)+"_W"+str(gridHeight)+"_N"+str(noAgent)+
              "_O"+str(noObs)+"_E"+str(totalEpisode)+"_Nw"+str(neighborWeights), "rb") as Pp:
            position = pickle.load(Pp)
        
        # with open("./RandomPosition/C"+str(0)+ "_L"+str(LoopVal)+
        #       "_H"+str(gridHeight)+"_W"+str(gridHeight)+"_N"+str(noAgent)+
        #       "_O"+str(noObs)+"_E"+str(totalEpisode)+"_Nw"+str(neighborWeights), "rb") as Pp:   # Unpickling
        #     position = pickle.load(Pp)

        aPosListTotal = position[countVal]
        if playMode['Agent'].lower() == 'random':
            self.aPosList = aPosListTotal[epochVal]
            
            for a in range(noAgent):
                self.state = self.width * self.aPosList[a][1] + self.aPosList[a][0]
                self.stateList.append(self.state)
                self.doneList.append('False')
                self.rewardList.append(0)


        # reset targets' position
        self.tPosList = []
        if playMode['Target'].lower() == 'random':
            for t in range(noTarget):
                tPosX = np.random.choice(self.sX)
                tPosY = np.random.choice(self.sY)
                self.tPosList.append([tPosX, tPosY])
        elif playMode['Target'].lower() == 'static':
            for t in range(noTarget):
                self.tPosList.append([(int(self.height/(t+1))-1), (int(self.width/(t+1))-1)])
        else:
            for t in range(noTarget):
                self.tPosList.append([self.height-1,self.width-1])

        # reset obstacles' position
        self.oPosList = []
        if playMode['Obstacle'].lower() == 'random':
            for o in range(noObs):
                oPosX = np.random.choice(self.sX)
                oPosY = np.random.choice(self.sY)
                if (([oPosX, oPosY] not in self.tPosList) and ([oPosX, oPosY] not in self.aPosList)):
                    self.oPosList.append([oPosX, oPosY])
                else:
                    oPosX = np.random.choice(self.sX)
                    oPosY = np.random.choice(self.sY)
                    self.oPosList.append([oPosX, oPosY])
        elif playMode['Obstacle'].lower() == 'static':
            for o in range(noObs):
                self.oPosList.append([(int(self.height/(o+1))-1), (int(self.width/(o+1))-2)])
        else:
            for o in range(noObs):
                self.oPosList.append([0,0])

        # reset freeways' position
        self.fPosList = []
        if playMode['Freeway'].lower() == 'random':
            for f in range(noFreeway):
                fPosX = np.random.choice(self.sX)
                fPosY = np.random.choice(self.sY)
                if (([fPosX, fPosY] not in self.tPosList) and ([fPosX, fPosY] not in self.aPosList) and ([fPosX, fPosY] not in self.oPosList)):
                    self.fPosList.append([fPosX, fPosY])
                else:
                    fPosX = np.random.choice(self.sX)
                    fPosY = np.random.choice(self.sY)
                    self.fPosList.append([fPosX, fPosY])
        elif playMode['Freeway'].lower() == 'static':
            for f in range(noFreeway):
                self.fPosList.append([(int(self.height/(f+1))-2), (int(self.width/(f+1))-1)])
        else:
            for f in range(noFreeway):
                self.fPosList.append([0,0])

        return self.tPosList, self.aPosList, self.stateList, self.rewardList, self.doneList, self.oPosList, self.fPosList, self.courierNumber
        # return self.state1, self.reward1, self.done1

    # take action
    def step(self, actionList, doneList, noTarget, noAgent, noObs, noFreeway,
        actionReward, obsReward, freewayReward, emptycellReward, hitwallReward, completedAgent, goalReward):
    # def step(self, action1):
        self.doneList = doneList
        nextStateList = []
        self.rewardList = []
        self.doneList = []

        actionReward = actionReward
        obsReward = obsReward #-1.5
        freewayReward = freewayReward #2
        emptycellReward = emptycellReward #1
        hitwallReward = hitwallReward #-0.5
        
        self.oPosList = []
        for o in range(noObs):
            oPosX = np.random.choice(self.sX)
            oPosY = np.random.choice(self.sY)
            if (([oPosX, oPosY] not in self.tPosList) and ([oPosX, oPosY] not in self.aPosList) and ([oPosX, oPosY] not in self.oPosList)):
                self.oPosList.append([oPosX, oPosY])
            else:
                oPosX = np.random.choice(self.sX)
                oPosY = np.random.choice(self.sY)
                self.oPosList.append([oPosX, oPosY])
                

        if self.courierNumber!= 0:
            self.fPosList = self.fPosList
        else:
            self.fPosList = []
        
       
        for a in range(noAgent):
            if a not in completedAgent:
                # ----------a=0, left------------------
                if actionList[a] == 0: # left
                    if self.aPosList[a][0] > 0:
                        self.aPosList[a][0] = self.aPosList[a][0]-1
                    elif (self.aPosList[a] in self.oPosList):
                        # self.aPosList[a][0] = self.aPosList[a][0]-1
                        actionReward = obsReward # if falls into obstacle, reward = -1.5
                    elif (self.aPosList[a] in self.fPosList):
                        # self.aPosList[a][0] = self.aPosList[a][0]-1
                        actionReward = freewayReward # if falls into freeway, reward = 2
                        self.courierNumber = self.courierNumber-1
                    else:
                        self.aPosList[a][0]
                        actionReward = hitwallReward # if hits a wall, reward = -0.5

                # ----------a=1, right------------------
                if actionList[a] == 1: # right
                    if self.aPosList[a][0] < (self.width - 1):
                        self.aPosList[a][0] = self.aPosList[a][0]+1
                    elif (self.aPosList[a] in self.oPosList):
                        # self.aPosList[a][0] = self.aPosList[a][0]+1
                        actionReward = obsReward # if falls into obstacle, reward = -1.5 
                    elif (self.aPosList[a] in self.fPosList):
                        # self.aPosList[a][0] = self.aPosList[a][0]-1
                        actionReward = freewayReward # if falls into freeway, reward = 2
                        self.courierNumber = self.courierNumber-1
                    else:
                        self.aPosList[a][0]
                        actionReward = hitwallReward # if hits a wall, reward = -0.5

                # ----------a=2, Up------------------
                if actionList[a] == 2: # up
                    if self.aPosList[a][1] > 0:
                        self.aPosList[a][1] = self.aPosList[a][1]-1
                    elif (self.aPosList[a] in self.oPosList):
                        # self.aPosList[a][1] = self.aPosList[a][1]-1
                        actionReward = obsReward # if falls into obstacle, reward = -1.5 
                    elif (self.aPosList[a] in self.fPosList):
                        # self.aPosList[a][0] = self.aPosList[a][0]-1
                        actionReward = freewayReward # if falls into freeway, reward = 2
                        self.courierNumber = self.courierNumber-1
                    else:
                        self.aPosList[a][1]
                        actionReward = hitwallReward # if hits a wall, reward = -0.5

                # ----------a=3, Down------------------
                if actionList[a] == 3: # down
                    if self.aPosList[a][1] < (self.height - 1):
                        self.aPosList[a][1] = self.aPosList[a][1]+1
                    elif (self.aPosList[a] in self.oPosList):
                        # self.aPosList[a][1] = self.aPosList[a][1]+1
                        actionReward = obsReward # if falls into obstacle, reward = -1.5
                    elif (self.aPosList[a] in self.fPosList):
                        # self.aPosList[a][0] = self.aPosList[a][0]-1
                        actionReward = freewayReward # if falls into freeway, reward = 2
                        self.courierNumber = self.courierNumber-1
                    else:
                        self.aPosList[a][1]
                        actionReward = hitwallReward # if hits a wall, reward = -0.5

                if (self.aPosList[a] in self.tPosList):
                    done = 'True'
                else:
                    done = 'False'

                nextState = self.width * self.aPosList[a][1] + self.aPosList[a][0]
                    
                if done == 'True':
                    reward = goalReward
                else:
                    reward = actionReward
            else:
                self.aPosList[a][0] = self.aPosList[a][0]
                self.aPosList[a][1] = self.aPosList[a][1]
                nextState = self.width * self.aPosList[a][1] + self.aPosList[a][0]
                done = 'True'
                reward = 0

            self.doneList.append([a, done])
            nextStateList.append(nextState)
            self.rewardList.append(reward)


        
        
        return nextStateList, self.rewardList, self.doneList, self.oPosList, self.courierNumber
        
        # return nextState1, self.reward1, self.done1

    # return a random action
    def randomAction(self):
        action = np.random.choice(self.actions)
        return action

    def neighbors(self, noAgent, aPosList, gridWidth, gridHeight, flag):
        if flag == 0:
            radius = math.ceil(np.sqrt((gridWidth*gridHeight)/noAgent))
        else:
            radius = gridWidth
        
        neighborZone = [radius, radius, radius, radius]
        self.aPosList = aPosList
        neighborDict = {}
        for a in range(noAgent):
            neighborDict[a] = []
        for a in range(noAgent):
            PlayerPos = aPosList[a]
            otherPlayerPos = aPosList[:]
            otherPlayerPos.remove(aPosList[a])

            NeighborPos = []
            
            upVal = [-u for u in range(neighborZone[2], 0, -1)]
            downVal = [d for d in range(neighborZone[3], 0, -1)]
            
            upDownList = [upVal,[0],downVal]
            upDownList = [item for sublist in upDownList for item in sublist]

            leftVal = [-l for l in range(neighborZone[0], 0, -1)]
            rightVal = [r for r in range(neighborZone[1], 0, -1)]
            leftRightList = [leftVal,[0],rightVal]
            leftRightList = [item for sublist in leftRightList for item in sublist]

            for l in upDownList:
                for m in leftRightList:
                    NeighborPos.append([PlayerPos[0]+l, PlayerPos[1]+m])


            for item in otherPlayerPos:
                if item in NeighborPos:
                    neighborDict[a].append(item)
                    # neighborDict[a].append(item[1])
        return neighborDict

    # display environment
    def render(self):
        for i in range(self.height):
            for j in range(self.width):
                if [i,j] in self.aPosList:
                    self.aPosIndex = self.aPosList.index([i,j])
                    print(" P"+str(self.aPosIndex)+" ", end='')
                elif [i,j] in self.tPosList:
                    self.tPosIndex = self.tPosList.index([i,j])
                    print(" T"+str(self.tPosIndex)+" ", end='')
                elif [i,j] in self.oPosList:
                    self.oPosIndex = self.oPosList.index([i,j])
                    print(" O"+str(self.oPosIndex)+" ", end='')
                elif [i,j] in self.fPosList:
                    self.fPosIndex = self.fPosList.index([i,j])
                    print(" F"+str(self.fPosIndex)+" ", end='')
                else:
                    print(" .  ", end='')
            print("")