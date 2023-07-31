#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# you need to run PeLPA.py before running this file.

import sys
import math
import pickle
import numpy as np
import pandas as pd

episodeNum = int(sys.argv[1])
sg_gap = int(sys.argv[2])
RewardGap = int(sys.argv[3])
convGap = int(sys.argv[4])
disGap = int(sys.argv[5])
genGap = int(sys.argv[6])
nnGap = int(sys.argv[7])
atnGap = int(sys.argv[8])
envType = sys.argv[9].lower()


#### Edit these lines according to your experiments and Graph generation preferences #################
# specifies framework names
frameworkName = ['RAMPART'] 

# look into 'Main/OutputFile/RAMPART.txt to fetch your desired file name' 
# place those file names sequentially (PeLPA) inside fName variable. 
# here, '1729b'/a2075'/'dc756' are some example file names


fName = [
        ['dc1d8']
        ]


# specify attackers' percentage here
AttackPercentage=['30']
######################################################################################################

BRNESList = []
AdhocTDList = []
DARLList = []
df = pd.DataFrame({})
df1 = pd.DataFrame({})
df2 = pd.DataFrame({})
df3 = pd.DataFrame({})
df4 = pd.DataFrame({})
df5 = pd.DataFrame({})
df6 = pd.DataFrame({})
df7 = pd.DataFrame({})
df8 = pd.DataFrame({})
df9 = pd.DataFrame({})
df10 = pd.DataFrame({})


def Asfunc(lst_gap, lst):
    lstL =[]
    lstAvg1 =[]
    lstAvg2 =[]
    
    for i in range(0,len(lst)):
        lstL.append(lst[i][0])
    lstAvg1 = [lstL[x:x+lst_gap] for x in range(0, len(lstL),lst_gap)]
    for i in range(0, len(lstAvg1)):
        lstAvg2.append(sum(lstAvg1[i])/len(lstAvg1[i]))
    return lstAvg2

def moving_average(data, window_size):
    return pd.Series(data).rolling(window=window_size).mean()

for i in range(len(fName)):
    for j in range(len(frameworkName)):
        fileName = fName[i][j]
        
        ###### SG ####################################################################################
        with open("./SG/"+str(envType)+"/"+str(fileName)+"_"+str(frameworkName[j])+"_Step", "rb") as Sp:
            stepVal = pickle.load(Sp)
        
        stepValAvg = [np.average(x) for x in zip(*stepVal)]
        stepValAvg = stepValAvg[0:episodeNum]
        stepValAvg = [stepValAvg[x:x+sg_gap] for x in range(0, len(stepValAvg),sg_gap)]
        stepValAvg = np.mean(np.array(stepValAvg), axis = 1)
        
        ###### Reward ####################################################################################
        with open("./Reward/"+str(envType)+"/"+str(fileName)+"_"+str(frameworkName[j])+"_Reward", "rb") as Rp:
            rewardVal = pickle.load(Rp)
        epochAvg = []
        for y in range(len(rewardVal[0])):
            epochAvg.append([np.average(z) for z in zip(*(rewardVal[x][y] for x in range(len(rewardVal))))])
            
        rewardValAvg = [list(x) for x in zip(*epochAvg)]
        listItem = []
        listItemAvg = []
        
        for p in range(episodeNum):
            for item in rewardValAvg:
                listItem.append(item[p])
            listItemAvg.append(sum(listItem)/len(listItem))
            listItem = []
            
        listItemAvg = listItemAvg[0:episodeNum]
        rewardValAvg0 = [listItemAvg[x:x+RewardGap] for x in range(0, len(listItemAvg),RewardGap)]
        rewardValAvg0 = np.mean(np.array(rewardValAvg0), axis = 1)
        
        
        
        ###### Convergence  ####################################################################################
        with open("./Convergence/"+str(envType)+"/"+str(fileName)+"_"+str(frameworkName[j])+"_convergence", "rb") as Cp:
            convergenceVal = pickle.load(Cp)
        convergenceAvg = [np.average(x) for x in zip(*convergenceVal)]
        convergenceAvg = convergenceAvg[0:episodeNum]
        convergenceAvg = [convergenceAvg[x:x+convGap] for x in range(0, len(convergenceAvg),convGap)]
        convergenceAvg = np.mean(np.array(convergenceAvg), axis = 1)

        ##### Generator & Discriminator Loss ##################################################################
        with open("./Loss/"+str(envType)+"/"+str(fileName)+"_"+str(frameworkName[j])+"_DisLoss", "rb") as Lp:
            dLoss = pickle.load(Lp)

        with open("./Loss/"+str(envType)+"/"+str(fileName)+"_"+str(frameworkName[j])+"_GenLoss", "rb") as Lp:
            gLoss = pickle.load(Lp)


        disLossAvgFinal = []
        disLossAvg = [dLoss[0][x:x+disGap] for x in range(0, len(dLoss[0]),disGap)]
        for item in disLossAvg:
            disLossAvgFinal.append(np.mean(np.array(item)))

        genLossAvgFinal = []
        genLossAvg = [gLoss[0][x:x+genGap] for x in range(0, len(gLoss[0]),genGap)]
        for item in genLossAvg:
            genLossAvgFinal.append(np.mean(np.array(item)))



        ###### Anomaly score #################################################################################
        with open("./AS/"+str(envType)+"/"+str(fileName)+"_"+str(frameworkName[j])+"_AS_nn", "rb") as ASp:   #Pickling
            nn = pickle.load(ASp) 
        with open("./AS/"+str(envType)+"/"+str(fileName)+"_"+str(frameworkName[j])+"_AS_atn", "rb") as ASp:   #Pickling
            atn = pickle.load(ASp) 

        nnP = Asfunc(nnGap, nn)
        atnP = Asfunc(atnGap, atn)
        malicious_advices = np.array(atnP)
        benign_advices = np.array(nnP)
        malicious_advices = malicious_advices[:len(malicious_advices)//10*10]
        benign_advices = benign_advices[:len(benign_advices)//10*10]
        malicious_advices_chunks = np.mean(malicious_advices.reshape(-1, 10), axis=1)
        benign_advices_chunks = np.mean(benign_advices.reshape(-1, 10), axis=1)
        # Convert the lists to pandas Series
        malicious_advices_series = pd.Series(malicious_advices)
        benign_advices_series = pd.Series(benign_advices)
        # Choose a window size for moving average
        window_size = 10
        # Calculate moving averages
        malicious_advices_ma = moving_average(malicious_advices_series, window_size)
        benign_advices_ma = moving_average(benign_advices_series, window_size)
        
        
        if frameworkName[j]=='RAMPART':
            df2["RAMPART"+str(AttackPercentage[i])]=convergenceAvg
            df["RAMPART"+str(AttackPercentage[i])]=stepValAvg
            df1["RAMPART"+str(AttackPercentage[i])]=rewardValAvg0
            df3["RAMPART"+str(AttackPercentage[i])]=disLossAvgFinal
            df4["RAMPART"+str(AttackPercentage[i])]=genLossAvgFinal
            df5["RAMPART"+str(AttackPercentage[i])+"_line_m"]=malicious_advices_ma
            df6["RAMPART"+str(AttackPercentage[i])+"_hist_m"]=malicious_advices_chunks
            df7["RAMPART"+str(AttackPercentage[i])+"_kde_cdf_m"]=malicious_advices
            df8["RAMPART"+str(AttackPercentage[i])+"_line_b"]=benign_advices_ma
            df9["RAMPART"+str(AttackPercentage[i])+"_hist_b"]=benign_advices_chunks
            df10["RAMPART"+str(AttackPercentage[i])+"_kde_cdf_b"]=benign_advices
        else:
            print('*Error in processing file*')
            
df.to_csv("./ProcessedOutput/ProcessedSG_"+str(envType)+".csv", index=False)
df1.to_csv("./ProcessedOutput/ProcessedReward_"+str(envType)+".csv", index=False)
df2.to_csv("./ProcessedOutput/ProcessedConvergence_"+str(envType)+".csv", index=False)
df3.to_csv("./ProcessedOutput/ProcessedDiscriminatorLoss_"+str(envType)+".csv", index=False)
df4.to_csv("./ProcessedOutput/ProcessedGeneratorLoss_"+str(envType)+".csv", index=False)
df5.to_csv("./ProcessedOutput/ProcessedAnomalyScore_line_m_"+str(envType)+".csv", index=False)
df6.to_csv("./ProcessedOutput/ProcessedAnomalyScore_hist_m_"+str(envType)+".csv", index=False)
df7.to_csv("./ProcessedOutput/ProcessedAnomalyScore_kde_cdf_m_"+str(envType)+".csv", index=False)
df8.to_csv("./ProcessedOutput/ProcessedAnomalyScore_line_b_"+str(envType)+".csv", index=False)
df9.to_csv("./ProcessedOutput/ProcessedAnomalyScore_hist_b_"+str(envType)+".csv", index=False)
df10.to_csv("./ProcessedOutput/ProcessedAnomalyScore_kde_cdf_b_"+str(envType)+".csv", index=False)
print("Processed files are ready inside the Main/ProcessedOutput folder")
        
            
        
        