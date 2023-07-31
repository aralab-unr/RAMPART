#!/usr/bin/env python
# coding: utf-8

import sys
import os
from PP_environment.environment import Env
from scipy.stats import truncnorm
import pandas as pd
import numpy as np
import random
import pickle
import uuid
import time
import math
import os
import copy
from operator import add, sub, mul
from termcolor import colored
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import matplotlib.animation as FuncAnimation
from IPython import display
from IPython.display import HTML
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Input, Embedding, Flatten, Concatenate
from tensorflow.keras.optimizers import Adam
from numpy.random import randn
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import plot_model

## starting main program
start = time.time()


gridHeightList = [int(sys.argv[1])]
gridWidthList = [int(sys.argv[1])]
noAgentList = [int(sys.argv[2])]
noObsList = [int(sys.argv[3])]
eList = [int(sys.argv[4])]
LoopVal = int(sys.argv[5]) # defines how many times the code will run
neighborWeightsList = [float(sys.argv[6])]
attackPercentage = [int(sys.argv[7])]
display = sys.argv[8]
sleep = float(sys.argv[9])
try:
    mode = sys.argv[10]
except:
    mode = 'random'
    
env_type=sys.argv[11].lower()

lossT = float(sys.argv[12])
epochB = int(sys.argv[13])
episodeL = int(sys.argv[14])
detectT = float(sys.argv[15])

if mode.lower()=='random':
    playModeList = {"Agent":'random', "Target":'static', "Obstacle":'random', "Freeway":'random'}
else:
    playModeList = {"Agent":'random', "Target":'static', "Obstacle":'static', "Freeway":'static'}
    
flag = 0 # flag = 0, neighbor zone enabled and flag = 1, neighbor zone disabled

noTarget = 1 # there is only one target
noFreeway = 1 # there is only one freeway/resting area
AttackerList = [random.sample(range(1,noAgentList[0]), math.ceil(noAgentList[0]*attackPercentage[0]/100))] # calculating attackers' list
print("Attackers: ", AttackerList)


fig, axs = plt.subplots((noAgentList[0])-len(AttackerList[0])+1,1, sharex=True)

 # reward and penalties
actionReward = 0
obsReward = -1.5
freewayReward = 0.5
emptycellReward = 0
hitwallReward = -0.5
goalReward = 10

min_v = obsReward+hitwallReward
max_v = goalReward+freewayReward+emptycellReward

# hyper-parameters
alpha = 0.1 # RL learning rate
varepsilon = 0.5 # privacy budget
degree = 10


### Laplace-based LDP mechanism
def Lap(randomlist, max_v, min_v, varepsilon, alpha):
    # print("actual value: ", randomlist)
    p_dataset = []
    b = (((max_v-min_v)*alpha)/varepsilon)
    for val in range(len(randomlist)):
        p_val = val + np.random.laplace(0, b)
        while ((p_val<min_v) or (p_val>max_v)):
            p_val = val + np.random.laplace(0, b)
        p_dataset.append(p_val)
    return p_dataset


### LDP-exploited attack mechanism
def attack(oldQAttacker, degree, min_v, max_v, varepsilon, alpha, adviseeQ):
    newQAttackerDegree =[]
    newQAttacker = []
    newQAttackerList = []
    b = ((max_v-min_v)*alpha)/varepsilon
    degreeVal = 0
    degreeFlag = True
    while degreeFlag==True:
        degreeVal += 1
        func = lambda k: ((2*b**2)/(k**2 - b**2)) - np.log(k**2) + np.log((k**2) - (b**2)) - degreeVal
        k_initial_guess = b+1
        p = fsolve(func, k_initial_guess)

        k1 = p[0]
        theta = 0
        miu_attacker = (((b**2)*(theta - (2*k1))) - (theta * (k1**2))) /((b**2) - (k1**2))

        for Q in oldQAttacker:
            noise = np.random.laplace(miu_attacker, b)
            AdvQ = Q + noise
            while ((AdvQ<min_v) or (AdvQ>max_v)):
                noise = np.random.laplace(miu_attacker, b)
                AdvQ = Q + noise
            newQAttackerDegree.append(AdvQ)
            
        if ((newQAttackerDegree[adviseeQ.index(max(adviseeQ))]<max(adviseeQ)) or (degreeVal>15)):
            degreeFlag = False
        else:
            pass
        item = newQAttackerDegree.copy()
        newQAttackerDegree = []
        
    print("Degree Val: ",degreeVal)
    return degreeVal, item


def GAN(n_states):
    n_states = n_states
    d_lossList= []
    g_lossList = []
    latent_dim = 10
    epochs = 100
    n_batch = 6

    # define the standalone discriminator model
    def define_discriminator(n_inputs=4):
        # State input
        state_input = Input(shape=(1,))
        state_embed = Embedding(n_states, 10)(state_input)
        state_embed = Flatten()(state_embed)

        # Q-values input
        q_values_input = Input(shape=(n_inputs,))

        # Concatenate state and q_values
        merged = Concatenate()([state_embed, q_values_input])

        # Hidden layers
        hidden1 = Dense(25, activation='relu')(merged)
        hidden2 = Dense(15, activation='relu')(hidden1)  # Updated hidden layer size

        # Output layer
        output = Dense(1, activation='sigmoid')(hidden2)

        model = Model([state_input, q_values_input], output)

        # compile model
        model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5), metrics=['accuracy'])
        return model


    # define the standalone generator model
    def define_generator(latent_dim, n_outputs=4):
        # State input
        state_input = Input(shape=(1,))
        state_embed = Embedding(n_states, 10)(state_input)
        state_embed = Flatten()(state_embed)

        # Latent space input
        latent_input = Input(shape=(latent_dim,))

        # Concatenate state and latent input
        merged = Concatenate()([state_embed, latent_input])

        # Hidden layer
        hidden1 = Dense(25, activation='relu')(merged)  # Change hidden layer size to 25

        # Output layer
        output = Dense(n_outputs, activation='linear')(hidden1)

        model = Model([state_input, latent_input], output)

        model.compile(loss='mean_absolute_error', optimizer=Adam(learning_rate=0.0002, beta_1=0.5), metrics=['mean_absolute_error'])

        return model


    def define_gan(generator, discriminator):
        # Get the generator input layers
        generator_state_input = generator.input[0]
        generator_latent_input = generator.input[1]

        # Generate fake samples
        generator_output = generator([generator_state_input, generator_latent_input])

        # Get the discriminator output from the generator
        discriminator_output = discriminator([generator_state_input, generator_output])

        # Combine the generator and discriminator into a single model
        model = Model([generator_state_input, generator_latent_input], discriminator_output)

        # Compile the model
        model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))

        return model
    
    discriminator = define_discriminator()
    # create the generator
    generator = define_generator(latent_dim)
    # create the gan
    gan_model = define_gan(generator, discriminator)
    
    generator.save('models/generator_model.h5')
    discriminator.save('models/discriminator_model.h5')
    gan_model.save('models/gan_model.h5')
    
    return generator, discriminator, gan_model, epochs, n_batch, n_states, d_lossList, g_lossList, latent_dim




def GAN_train(generator, discriminator, gan_model, latent_dim, n_epochs, n_batch, noq, n_states, d_lossList, g_lossList, epochB,  lossT):

    # use the generator to generate n fake examples, with class labels
    def generate_fake_samples(generator, latent_dim, half_batch, fake_states):
        # generate points in the latent space
        x_input = generate_latent_points(latent_dim, half_batch)
        # predict outputs
        X = generator.predict([np.array(fake_states), x_input], verbose=0)
        # create class labels
        y = np.zeros((half_batch, 1))
        return X, y
    

    # generate points in latent space as input for the generator
    def generate_latent_points(latent_dim, half_batch):
        # generate points in the latent space
        x_input = randn(latent_dim * half_batch)
        # reshape into a batch of inputs for the network
        x_input = x_input.reshape(half_batch, latent_dim)
        return x_input


    def train(g_model, d_model, gan_model, latent_dim, neighborsOldQ, lossT, n_epochs=epochB,  n_batch=6, n_eval=100):     
        for GAN_i in range(n_epochs):
            # prepare real samples
            population = list(set(nonEmptyState))
            real_states = random.sample(population, half_batch)

            real_q_values = np.concatenate([np.expand_dims(neighborsOldQ[state][np.random.randint(0, len(neighborsOldQ[state]))], axis=0) for state in real_states], axis=0)
            y_real = np.ones((half_batch, 1))

            # prepare fake examples using generator model
            fake_states = np.random.randint(0, n_states, half_batch)
            x_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch,fake_states)

            real_states = np.array(real_states)
            
            # update discriminator
            d_loss_real = d_model.train_on_batch([real_states, real_q_values], y_real)
            d_loss_fake = d_model.train_on_batch([fake_states, x_fake], y_fake)
            d_loss = 0.5 * np.add(d_loss_real[0], d_loss_fake[0])
            d_lossList.append(d_loss)

            # prepare points in latent space as input for the generator
            gan_states = np.random.randint(0, n_states, n_batch)
            x_gan = generate_latent_points(latent_dim, n_batch)

            # create inverted labels for the fake samples
            y_gan = np.ones((n_batch, 1))

            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch([gan_states, x_gan], y_gan)
            g_lossList.append(g_loss)

            # evaluate the model every n_eval epochs
            if (GAN_i+1) % n_eval == 0:
                print(f"Eval step: {GAN_i+1}, [D loss: {d_loss}] [G loss: {g_loss}]")

                eval_state = random.sample(population, 1)
                real_data = neighborsOldQ[eval_state[0]]
                fake_states = np.array(eval_state)
                pred_data = generate_fake_samples(generator, latent_dim, 1,fake_states)[0]
            GANflag = 1
            if d_loss > lossT:
                print("GAN training stopped at epoch: "+str(GAN_i)+" since discriminator's loss exceeds the loss threshold!")
                break
            


        return GANflag, d_lossList, g_lossList
    
    neighborsOldQ = noq
    
    # train model
    GANflag, d_lossList, g_lossList = train(generator, discriminator, gan_model, latent_dim, neighborsOldQ, lossT, n_epochs, n_batch)
    return GANflag, d_lossList, g_lossList, generator, discriminator, latent_dim


def detect_anomalies(generator, discriminator, latent_dim, count, neighbor_data, state):
    test_state = np.array([state])
    #  use the generator to generate n fake examples, with class labels
    def generate_fake_samples(generator, latent_dim, count, fake_states):
        # generate points in the latent space
        x_input = generate_latent_points(latent_dim, count)
        # predict outputs
#         X = generator.predict([np.random.randint(0, n_states, n), x_input], verbose=0)
        X = generator.predict([np.array(fake_states), x_input], verbose=0)
        # create class labels
        y = np.zeros((count, 1))
        return X, y

    # generate points in latent space as input for the generator
    def generate_latent_points(latent_dim, count):
        # generate points in the latent space
        x_input = randn(latent_dim * count)
        # reshape into a batch of inputs for the network
        x_input = x_input.reshape(count, latent_dim)
        return x_input
    
    
    test_state = np.array([state])
    auth_neighbor = discriminator.predict([test_state, neighbor_data])
    print("neighbor_synthetic: ", auth_neighbor)
    
    return auth_neighbor


 # initializing lists for calculating and saving convergence values
diffAvg1 = []
diffAvg2 = []
diffAvg3 = []
diffAvg4 = []
diffAvg5 = []
diffAvg4_0 = []
n_states = gridHeightList[0]*gridWidthList[0]
noq = {i:[] for i in range(n_states)}
d_lossList= []
g_lossList = []
attack_neighborL = []
noAttack_neighborL = []
noqList = []
GAN_trainFlag = 0
GANtrainCount =0

discardedAdvice = 0
keptAdvice = 0


# Main Loop ############################################################
for CriteriaVal in range(len(gridWidthList)):
    print("##################### Criteria Value: "+str(CriteriaVal)+" #######################\n")
    Attacker = AttackerList[CriteriaVal]
    Behb_tot = [100000 for tk in range(noAgentList[CriteriaVal])] # advisee's budget for seeking advice during experience harvesting (EH)
    Besb_tot = [10000 for pk in range(noAgentList[CriteriaVal])] # advisors' budget for seeking advice during experience giving (EG)
    fileName = str(uuid.uuid4())[:5] # initializing unique filename for storing learning outcomes
    stepsListFinal = []
    stepAgentListFinal = []
    rewards_all_episodesFinal = []
    qtableListFinal = []
    diffAvg5 = []
    for countVal in range(LoopVal):
        gridWidth = gridWidthList[CriteriaVal]
        gridHeight = gridHeightList[CriteriaVal]
        playMode = playModeList
        noAgent = noAgentList[CriteriaVal]
        noObs = noObsList[CriteriaVal]
        neighborWeights = neighborWeightsList[CriteriaVal]

        ## initialize varaibles
        qtableList = []
        aPosList = []
        stateList = []
        rewardList = []
        doneList = []
        actionList = []
        nextStateList = []
        rewards_all_episodes = []
        visitCount = []

        ## Check if no of elements greater than the state space or not
        if (noAgent+noTarget+noObs+noFreeway)>= (gridHeight * gridWidth):
            print("Total number of elements (agents, targets, obstacles) exceeds grid position")
        else:
            # building environment
            env = Env(gridHeight, gridWidth, playMode, noTarget, noAgent, noObs, noFreeway)
            print('-------Initial Environment---------\n')
            print("\n")

        ## for each agent, initializing a Q-table with random Q-values
        for a in range(noAgent):
            qtableList.append(np.random.rand(env.stateCount, env.actionCount).tolist())
        a = 0
            
        for a in range(noAgent):
            noqList.append(noq)
        a = 0
        
        genList = []
        disList = []
        ganList = []
        epochList = []
        batchList = []
        stList = []
        genLossList = []
        disLossList = []
        latentList = []
        
        for a in range(noAgent):
            genList = [[]for a in range(noAgent)]
            disList = [[]for a in range(noAgent)]
            ganList = [[]for a in range(noAgent)]
            epochList = [[]for a in range(noAgent)]
            batchList = [[]for a in range(noAgent)]
            stList = [[]for a in range(noAgent)]
            genLossList = [[]for a in range(noAgent)]
            disLossList = [[]for a in range(noAgent)]
            latentList = [[]for a in range(noAgent)]
            GANtrainCountL = [[0]for a in range(noAgent)]
        a = 0
            
        
        ## for each agent, initializing GAN models
        for a in range(noAgent):
            generator, discriminator, gan_model, n_epochs, n_batch, n_states, d_lossList, g_lossList, latent_dim = GAN(n_states)
            genList[a]=[generator]
            disList[a]=[discriminator]
            ganList[a]=[gan_model]
            epochList[a]=[n_epochs]
            batchList[a]=[n_batch]
            stList[a]=[n_states]
            genLossList[a]=g_lossList
            disLossList[a]=d_lossList
            latentList[a]=[latent_dim]
            
            generator.save('models/generator_model_'+str(a)+'.h5')
            discriminator.save('models/discriminator_model_'+str(a)+'.h5')
            gan_model.save('models/gan_model_'+str(a)+'.h5')
        a = 0

        ## hyperparameters
        totalEpisode = eList[CriteriaVal]
        gamma = 0.8 # discount factor
        epsilon = 0.08 #0.08 #exploration-exploitation
        intEpsilon = epsilon
        decay = 0.1 # decay of exploration-exploitation over episode
        stepsList = []
        alpha = 0.1 #learning rate

        
        ## initialize visit count for each state
        for i in range(noAgent):
            visitCount.append([0 for xV in range((gridWidth*gridHeight))])
            
        ## initialize current experience harvesting budget (EHB) and current experience sharing budget (ESB)
        Behb = Behb_tot.copy()
        Besb = Besb_tot.copy()
          
        stepAgentList = [[] for xS in range(noAgent)]    
        degreeValFinal = []  
        
        
        
        ## TRAINING loop---------#####------------------#####---------
        for epNum in range(totalEpisode):
            GANtrainCount = all(all(value > episodeL for value in nested_list) for nested_list in GANtrainCountL)
            if GANtrainCount==False and GAN_trainFlag!=1:
                print("GAN episode count (for all AVs): ", GANtrainCountL, '\n')
                for a in range(noAgent):
                    # determine half the size of one batch, for updating the discriminator
                    half_batch = int(batchList[a][0] / 2)
                    # manually enumerate epochs
                    nonEmptyState = []
                    for state in range(0,stList[a][0]):
                        if ((len(noqList[a][state])!=0) and (state not in nonEmptyState)):
                            nonEmptyState.append(state)

                    if len(nonEmptyState)>=half_batch: 
                        print("\nEpisode "+str(epNum)+": Agent-"+str(a)+" | GAN train started")
                        GANflag, d_lossList, g_lossList, generator, discriminator, latent_dim = GAN_train(genList[a][0], disList[a][0],
                                                                                                              ganList[a][0], latentList[a][0],
                                                                                                              epochList[a][0], batchList[a][0],
                                                                                                              noqList[a], stList[a][0], disLossList[a],
                                                                                                              genLossList[a], epochB, lossT)

                        # Check if the last value of g_lossList is lower than the previous value
                        if len(g_lossList) > 1 and d_lossList[-1] > d_lossList[-2] and g_lossList[-1] < g_lossList[-2] and g_lossList[-1]!=-1:
                            # Save the models
                            generator.save('models/generator_model_'+str(a)+'.h5')
                            discriminator.save('models/discriminator_model_'+str(a)+'.h5')
                            gan_model.save('models/gan_model_'+str(a)+'.h5')
                        else:
                            # Load the previous models
                            generator = load_model('models/generator_model_'+str(a)+'.h5')
                            discriminator = load_model('models/discriminator_model_'+str(a)+'.h5')
                            gan_model = load_model('models/gan_model_'+str(a)+'.h5')

                        disLossList[a] = d_lossList
                        genLossList[a] = g_lossList
                        genList[a] = [generator]
                        disList[a] = [discriminator]
                        latentList[a] = [latent_dim]
                        print("Train phase complete!")
                        GANtrainCountL[a] = [GANtrainCountL[a][0]+1]
                    else:
                        continue
                a = 0
                    

            
            degreeValListEp =[[] for xD in range(noAgent)]
            print("epoch #", epNum+1, "/", totalEpisode)
            tPosList, aPosList, stateList, rewardList, doneList, oPosList, fPosList, courierNumber = env.reset(playMode, noTarget, noAgent, noObs,
                                                                       noFreeway, gridWidth, gridHeight, epNum, CriteriaVal,countVal,neighborWeights,totalEpisode,LoopVal)
            rewards_current_episode =[0 for xR in range(noAgent)]
            doneList = [[a,'False'] for xDl in range(noAgent)]
            
            # render environment at the begining of every episode
            print("\n--------------Episode: ", epNum+1, " started----------------\n")            
            steps = 0
            completedAgent = []
            stepAgent = [0 for xSa in range(noAgent)]
            
            # uncomment only one line from below three lines according to your preference
            while [0, 'True'] not in doneList: # ends when agent0 reaches goal
#             while any('False' in sl for sl in doneList): # ends when all agents reach goal
#             while not any('True' in sl for sl in doneList): # ends when any agent reaches goal

                actionList = []
                if steps>(gridWidth*100):
                    break # break out of the episode if number of steps is too large to reach the goal.
                else:
                    steps +=1
                    
                ## find out neighbors starts---------------------------------------------------
                neighborDict = env.neighbors(noAgent, aPosList, gridWidth, gridHeight, flag)  
                neighborPosList = []
                for a in range(noAgent):
                    neighborsPrint = []
                    indNeighbor = []
                    for player in neighborDict[a]:
                        if a != aPosList.index(player):
                            indNeighbor.append(aPosList.index(player))
                        uniqueIndNeighbor = [*set(indNeighbor)]
                    neighborPosList.append(uniqueIndNeighbor)
                    uniqueIndNeighbor = []
                a = 0

                ## find out neighbors ends---------------------------------------------------
                
                ## find which agents have completed
                completedAgent = [xC for xC, yC in enumerate(doneList) if yC[1]=='True']
                
                for a in range(noAgent):
                    if ((a in completedAgent) and (stepAgent[a]==0)):
                        stepAgent[a] = steps
                a = 0
                
                ## update visit count for this state and every agent
                for a in range(noAgent):
                    visitCount[a][stateList[a]] += 1
                a = 0
                
                # Experience harvesting (EH) and Experience Giving (EG) phase
                for a in range(noAgent):
                    ## calculate Pehc (experience harvesting confidence) based on visit count and budget. 
                    # If visit count is too high (i.e., >100000) or too low (<100) for any episode, set experience harvesting confidence to low (i.e., will not seek for advice)
                    if ((visitCount[a][stateList[a]]< 100) or (visitCount[a][stateList[a]]> 100000)):
                        Pehc = 0
                    else:
                        Pehc = (1/np.sqrt(visitCount[a][stateList[a]])) * (np.sqrt(Behb[a]/Behb_tot[a]))
                    
                    if ((Pehc > 0) and (Pehc < 0.1)) :
                        Behb[a] = Behb[a]-1
                        QNeighbor  = []
                        if a not in completedAgent:
                            neighborsOldQ = 0
                            neighborsOldQList = []
                            selfOldQ = qtableList[a][stateList[a]]
                            if neighborPosList[a] !=[]:  #if not empty list
                                for n in neighborPosList[a]:
                                    ## calculate Pesc (experience sharing confidence) based on visit count and budget
                                    if (visitCount[n][stateList[a]]> visitCount[a][stateList[a]]):
                                        Pesc = (1-(1/np.sqrt(visitCount[n][stateList[a]]))) * (np.sqrt(Besb[n]/Besb_tot[n]))
                                    else:
                                        Pesc = 0
                                    ## if experience sharing confidence is high, give advice
                                    if Pesc > 0:
                                        
                                        Besb[n] = Besb[n]-1
                                        
                                        # incorporating LDP
                                        noisyQ = Lap(qtableList[n][stateList[a]], max_v, min_v, varepsilon, alpha)
                                        neighborsOldQ = noisyQ

                                        #### Attacking (if any attacker presents)
                                        ## if the neighbor is an attacker, proceed for attack
                                        if ((n in Attacker) and GANtrainCount==True):
                                            GAN_trainFlag=1
                                            # if the agent himself is not an attacker, proceed for attack
                                            if a not in Attacker:
                                                oldQAttacker = qtableList[n][stateList[a]].copy()
                                                degreeVal, neighborsOldQ = attack(oldQAttacker, degree, min_v, max_v, varepsilon, alpha, selfOldQ)
                                                print(colored("Honest", 'green') +" Advisee, P"+str(a)+" is receiving"+
                                                      colored(" Malicious", 'red', attrs=['reverse'])+
                                                      " advice from malicious Advisor, P"+str(n)+" at step: "+str(steps))
                                                degreeValListEp[n].append(degreeVal)


                                    
                                                ## GAN anomaly detection for malicious data
                                                attack_neighbor = detect_anomalies(genList[a][0],
                                                                                 disList[a][0],
                                                                                 latentList[a][0],
                                                                                 1,
                                                                                 np.array([neighborsOldQ]),
                                                                                 stateList[a])

                                                if attack_neighbor<=detectT:
                                                    neighborsOldQ = noisyQ # discard the advice
                                                    discardedAdvice +=1
                                                else:
                                                    neighborsOldQ = neighborsOldQ # keep the advice
                                                    keptAdvice +=1

                                                attack_neighborL.append(attack_neighbor)

                                            # if the advisee is an attacker, send him the non-malicious advice
                                            else:
                                                print(colored("Malicious", 'red')+" Advisee, P"+str(a)+" is receiving"+
                                                      colored(" Benign", 'green', attrs=['reverse'])+
                                                      " advice from malicious Advisor, P"+str(n)+" at step: "+str(steps))
                                                neighborsOldQ = neighborsOldQ

                                            # saving all the neighbors advice (malicious+non-malicious) in a list
                                            noqList[a][stateList[a]].append(neighborsOldQ) #noq means neighbors old Q

                                        # if the neighbor is NOT an attacker, DO NOT proceed for attack
                                        else:
                                            neighborsOldQ = neighborsOldQ
                                            print("Advisee, P"+str(a)+" is receiving"+
                                                      colored(" Benign", 'green', attrs=['reverse'])+
                                                      " advice from Advisor, P"+str(n)+" at step: "+str(steps))
                                            
                                            if GANtrainCount==True:
                                              ## GAN anomaly detection for non-malicious data
                                                noAttack_neighbor = detect_anomalies(genList[a][0],
                                                                                     disList[a][0],
                                                                                     latentList[a][0],
                                                                                     1,
                                                                                     np.array([neighborsOldQ]),
                                                                                     stateList[a])
                                                if noAttack_neighbor<=detectT:
                                                    neighborsOldQ = noisyQ # discard the advice
                                                    discardedAdvice +=1
                                                else:
                                                    neighborsOldQ = neighborsOldQ # keep the advice
                                                    keptAdvice +=1

                                                noAttack_neighborL.append(noAttack_neighbor)

                                            # saving all the neighbors advice (non-malicious) in a list
                                            noqList[a][stateList[a]].append(neighborsOldQ)

                                        # saving all the neighbors advice (attack or non-attack) in a list
                                        neighborsOldQList.append(neighborsOldQ)
                                    
                                    ## if experience sharing confidence is low, DO NOT give advice
                                    else:
                                        neighborsOldQ = []
                                        neighborsOldQList.append(neighborsOldQ)
                                
                                # combining neighbors expereince
                                if any(neighborsOldQList):
                                    for action in range(4): # here 4 stands for four different actions
                                        elem = [item[action] for item in neighborsOldQList if item!=[]]
                                        
                                        # selecting the most appropiate advice
                                        QNeighbor.append(np.mean(elem))
                                            
                                    # Weighted expereince aggregation
                                    qtableList[a][stateList[a]] = [sum(xt) for xt in zip([op * neighborWeights for op in selfOldQ], 
                                                     [oq * (1-neighborWeights) for oq in QNeighbor])]
                                else:
                                    qtableList[a][stateList[a]] = selfOldQ
                  
                
                # 1. select best action
                if np.random.uniform() < epsilon:
                    for a in range(noAgent):
                        actionList.append(env.randomAction())
                else:
                    for a in range(noAgent):
                        actionList.append(qtableList[a][stateList[a]].index(max(qtableList[a][stateList[a]])))
                        
                soqList = []   
                for a in range(noAgent):
                    soq = copy.deepcopy(qtableList[a])  #soq means self old Q
                    soqList.append(soq)
                
                # 2. take the action and observe next state & reward
                nextStateList, rewardList, doneList, oPosList, courierNumber = env.step(actionList, doneList, noTarget, noAgent, noObs, noFreeway,
                                                               actionReward, obsReward, freewayReward, emptycellReward,
                                                               hitwallReward, completedAgent, goalReward)

                # 3. Calculate self Q-value
                for a in range(noAgent):
                    if a not in completedAgent:
                        qtableList[a][stateList[a]][actionList[a]] = ((qtableList[a][stateList[a]][actionList[a]] * (1 - alpha)) + (alpha * (rewardList[a] + gamma * max(qtableList[a][nextStateList[a]]))))
                        rewards_current_episode[a] += rewardList[a]
                        stateList[a] = nextStateList[a]
                    else:
                        qtableList[a][stateList[a]][actionList[a]] = qtableList[a][stateList[a]][actionList[a]]
                        rewards_current_episode[a] += rewardList[a]
                        stateList[a] = nextStateList[a]

                snqList = []
                for a in range(noAgent):
                    snq = copy.deepcopy(qtableList[a]) # snq means self new Q
                    snqList.append(snq)
                
                # calcuating \Delta Q for convergence analysis
                for p in range(len(soq)):
                    for q in range(len(soq[p])):
                        diff = abs(soqList[0][p][q] - snqList[0][p][q])
                        diffAvg1.append(diff)
                    diffAvg2.append(sum(diffAvg1)/len(diffAvg1))
                    diffAvg1 = []
                diffAvg3.append(sum(diffAvg2)/len(diffAvg2))
                diffAvg2 = []
                
            
            degreeValFinal.append(degreeValListEp)
            degreeValListEp=[]
            diffAvg4.append(sum(diffAvg3)/len(diffAvg3))
            diffAvg3 = []
            
            
            
            epsilon -= decay*epsilon # decaying exploration-exploitation probability for future episodes
            
            stepsList.append(steps)
            rewards_all_episodes.append(rewards_current_episode)
            print("Done in", steps, "steps".format(steps))
            time.sleep(sleep)
            
            color = ['red','cyan','black', 'green', 'magenta', 'orange', 'yellow', 
                     'red','cyan','black', 'green', 'magenta', 'orange', 'yellow',
                    'red','cyan','black', 'green', 'magenta', 'orange', 'yellow']
            stepAgent[stepAgent.index(0)]= steps
            axsCount = 0
            for a in range(noAgent):
                stepAgentList[a].append(stepAgent[a])
                if a not in Attacker:
                    axs[axsCount].plot(stepAgentList[a], marker='.', color=color[axsCount])
                    axs[axsCount].set_xticks([i for i in range(len(stepAgentList[a]))][-1:])
                    axs[axsCount].set_ylabel('P'+str(a))
                    axs[axsCount].set_ylim(0,500)
                    axs[axsCount].grid()
                    axsCount = axsCount+1
            axs[0].set_title("Attackers are: "+"P"+str(Attacker))
            axs[(noAgent-len(Attacker))].plot(stepsList, marker='x', color='blue')
            axs[(noAgent-len(Attacker))].set_xticks([i for i in range(len(stepsList))][-1:])
            axs[(noAgent-len(Attacker))].set_ylabel('All')
            axs[(noAgent-len(Attacker))].grid()
            axs[(noAgent-len(Attacker))].set_ylim(0,500)

        stepsListFinal.append(stepsList)
        stepsList = []
        rewards_all_episodesFinal.append(rewards_all_episodes)
        rewards_all_episodes = []
        qtableListFinal.append(qtableList)
        qtableList = []
        diffAvg5.append(diffAvg4)
        diffAvg4 = []
        diffAvg4_0 =[]
    
    
    end = time.time()
    total_time = end-start
    print("Total Time taken: ",total_time) 
    
env_type='low'

dvf = degreeValFinal
with open("./SG/"+str(env_type)+"/"+str(fileName)+"_RAMPART_DegreeVal", "wb") as Sd:   #Pickling
    pickle.dump(dvf, Sd)

sa = stepAgentList
with open("./SG/"+str(env_type)+"/"+str(fileName)+"_RAMPART_Step_AgentWise", "wb") as Spa:   #Pickling
    pickle.dump(sa, Spa)

s = stepsListFinal
with open("./SG/"+str(env_type)+"/"+str(fileName)+"_RAMPART_Step", "wb") as Sp:   #Pickling
    pickle.dump(s, Sp)

r = rewards_all_episodesFinal
with open("./Reward/"+str(env_type)+"/"+str(fileName)+"_RAMPART_Reward", "wb") as Rp:   #Pickling
    pickle.dump(r, Rp)

q = qtableListFinal
with open("./Qtable/"+str(env_type)+"/"+str(fileName)+"_RAMPART_Qtable", "wb") as Qp:   #Pickling
    pickle.dump(q, Qp)

c = diffAvg5
with open("./Convergence/"+str(env_type)+"/"+str(fileName)+"_RAMPARTconvergence", "wb") as Cp:   #Pickling
    pickle.dump(c, Cp)


t = total_time
with open("./TG/"+str(env_type)+"/"+str(fileName)+"_RAMPART_Time", "wb") as Tp:   #Pickling
    pickle.dump(t, Tp)   
    
l = disLossList
with open("./Loss/"+str(env_type)+"/"+str(fileName)+"_RAMPART_DisLoss", "wb") as Lp:   #Pickling
    pickle.dump(l, Lp)

l = genLossList
with open("./Loss/"+str(env_type)+"/"+str(fileName)+"_RAMPART_GenLoss", "wb") as Lp:   #Pickling
    pickle.dump(l, Lp)


nn =[]
for ix in range(0, len(noAttack_neighborL)):
    nn.append(noAttack_neighborL[ix][0])
    
atn =[]
for iy in range(0, len(attack_neighborL)):
    atn.append(attack_neighborL[iy][0])
    
AS = nn
with open("./AS/"+str(env_type)+"/"+str(fileName)+"_RAMPART_AS_nn", "wb") as ASp:   #Pickling
    pickle.dump(AS, ASp)
    
AS = atn
with open("./AS/"+str(env_type)+"/"+str(fileName)+"_RAMPART_AS_atn", "wb") as ASp:   #Pickling
    pickle.dump(AS, ASp)
    



with open("./OutputFile/RAMPART.txt", "a") as myfile:
    myfile.write("FileName: "+str(fileName)+" : RAMPART, Time taken: "+str(total_time)+"\n | gridWidth: "+str(gridWidth)+" | gridHeight: "+str(gridHeight)+
                " | playMode: "+str(playMode)+" | noTarget: "+str(noTarget)+" | noAgent: "+str(noAgent)+
                " | noObs: "+str(noObs)+" | noFreeway: "+str(noFreeway)+
                " | neighborWeights: "+str(neighborWeights)+" | totalEpisode: "+str(totalEpisode)+" | gamma: "+str(gamma)+
                " | epsilon: "+str(intEpsilon)+" | decay: "+str(decay)+" | alpha: "+str(alpha)+
                " | obsReward: "+str(obsReward)+" | freewayReward: "+str(freewayReward)+" | emptycellReward: "+str(emptycellReward)+
                " | hitwallReward: "+str(hitwallReward)+" | Attacker: "+str(Attacker)+" | Notes: "+str("LDP+Attack+RAMPART")+"\n\n\n")   


