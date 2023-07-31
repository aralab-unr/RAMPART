#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import pyplot as plt, ticker as mticker
from matplotlib.ticker import StrMethodFormatter, NullFormatter
import seaborn as sns
import matplotlib.transforms as mtransforms

sns.set_theme(context = "notebook",style="ticks", palette='bright')
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["figure.autolayout"] = False
plt.rcParams["axes.labelsize"] = 15
plt.rcParams["axes.titlesize"] = 15
plt.rcParams["xtick.labelsize"] = 15
plt.rcParams["ytick.labelsize"] = 15
plt.rcParams["font.size"] = 15
plt.rcParams["lines.linewidth"] = 0.5
plt.rcParams["lines.markersize"] = 8
plt.rcParams["legend.fontsize"] = 15
plt.rcParams['grid.color'] = "#949292"
plt.rcParams['font.sans-serif'] = "Times New Roman"
plt.rcParams['font.family'] = "sans-serif"
legend_properties = {'weight':'bold'}
plt.rcParams['mathtext.fontset'] = 'cm'

fig = plt.figure(constrained_layout=False,figsize=(8,5))
gs= GridSpec(1, 1, figure=fig, wspace=0.1, hspace = 0.3)
ax0 = plt.subplot(gs.new_subplotspec((0, 0)))


axList = [[ax0]]

ax0.set_xlabel('Episodes, '+str(r"$e$"))
ax0.set_ylabel('Avg. reward, '+str(r"$\bar Φ$"))
labelList = ['(a)']

episodeNum = int(sys.argv[1])
RewardGap = int(sys.argv[2])
envType = sys.argv[3].lower()

x2 = list(np.linspace(10,episodeNum+10,int(episodeNum/RewardGap), endpoint=False)) # for reward

#### Edit these lines according to your experiments and Graph generation preferences #################
frameworkName = ['RAMPART']
AttackPercentage=['30']
trafficType = ['Low']
lineStyle = ['s-','v-','o-','^-.','*--','s-']
color = ['black','cyan','black','magenta','blue','green']
markerFace = ['none','cyan','none','magenta','blue','green']
markerEdge = ["black","black","black","black","black","black"]
legend = ['RAMPART']
######################################################################################################

## Make sure path exists for this one line of code #################################
df1 = pd.read_csv("./ProcessedOutput/ProcessedReward_"+str(envType)+".csv", delimiter=',')
################################################################################


### Reward #####
for i in range(len(AttackPercentage)):
	for j in range(len(frameworkName)):
		axList[j][0].plot(x2, df1[str(frameworkName[j])+str(AttackPercentage[i])], color = color[i]
                    # lineStyle[i], markerfacecolor=markerFace[i],
		                              # markeredgecolor=markerEdge[i], label = attackerNo[i]
		                              )
		axList[j][0].set_title(str(envType), loc="right" )
		# axList[j][0].legend(loc ='upper right', #bbox_to_anchor=(0.8, 0.5), #0.25,-1,
		#           ncol=1, fancybox=True, shadow=True,
		#            borderpad=0.2,labelspacing=0.2, handlelength=0.9,columnspacing=0.8,handletextpad=0.3)
		axList[j][0].tick_params(axis='x')
		axList[j][0].tick_params(axis='y')

        
		trans = mtransforms.ScaledTranslation(-1/72, 1/72, fig.dpi_scale_trans)
		axList[j][0].text(0.5, -0.35, labelList[j], transform=axList[j][0].transAxes + trans)


## Make sure path exists for this one line of code #################################
plt.savefig('./Reward_'+str(envType)+'.pdf',bbox_inches='tight') 
################################################################################
plt.show()


