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
plt.rcParams["lines.markersize"] = 0
plt.rcParams["legend.fontsize"] = 15
plt.rcParams['grid.color'] = "#949292"
plt.rcParams['font.sans-serif'] = "Times New Roman"
plt.rcParams['font.family'] = "sans-serif"
legend_properties = {'weight':'bold'}
plt.rcParams['mathtext.fontset'] = 'cm'


envType = sys.argv[1].lower()
window_size = 10

# #### Edit these lines according to your experiments and Graph generation preferences #################
frameworkName = ['RAMPART']
AttackPercentage=['30']
trafficType = ['Low']
lineStyle = ['s-','o-','d-','^-.','*--','s-']
lossType = ['Discriminator loss', 'Generator Loss']
color = ['red','aqua','black','magenta','blue','green']
markerFace = ['white','white','white','magenta','blue','green']
markerEdge = ["red","aqua","black","black","black","black"]
legend = ['RAMPART']
# ######################################################################################################

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20,3)) 

df1 = pd.read_csv("./ProcessedOutput/ProcessedAnomalyScore_line_m_"+str(envType)+".csv", delimiter=',')
df2 = pd.read_csv("./ProcessedOutput/ProcessedAnomalyScore_hist_m_"+str(envType)+".csv", delimiter=',')
df3 = pd.read_csv("./ProcessedOutput/ProcessedAnomalyScore_kde_cdf_m_"+str(envType)+".csv", delimiter=',')
df4 = pd.read_csv("./ProcessedOutput/ProcessedAnomalyScore_line_b_"+str(envType)+".csv", delimiter=',')
df5 = pd.read_csv("./ProcessedOutput/ProcessedAnomalyScore_hist_b_"+str(envType)+".csv", delimiter=',')
df6 = pd.read_csv("./ProcessedOutput/ProcessedAnomalyScore_kde_cdf_b_"+str(envType)+".csv", delimiter=',')


# Generate a line plot for the moving averages
ax1.plot(df1[str(frameworkName[0])+str(AttackPercentage[0])+"_line_m"], label='Malicious Advices', alpha=1, color='r', marker='o', 
         markeredgecolor='black', markerfacecolor='none')
ax1.plot(df4[str(frameworkName[0])+str(AttackPercentage[0])+"_line_b"], label='Benign Advices', alpha=1, color='g', marker='v', 
         markeredgecolor='black', markerfacecolor='none')
ax1.set_title('Moving Avg. (window size = {})'.format(window_size))
ax1.set_xlabel('Advice Chunk')
ax1.set_ylabel('Anomaly Score')
# ax1.legend()
ax1.grid(True)

# Histogram plot
ax2.hist([df2[str(frameworkName[0])+str(AttackPercentage[0])+"_hist_m"], df5[str(frameworkName[0])+str(AttackPercentage[0])+"_hist_b"]], bins=10, density=True,
         label=['Malicious Advices', 'Benign Advices'], alpha=0.6, color=['r','g'])
ax2.set_title('Freq. Dist.')
ax2.set_xlabel('Anomaly Score')
ax2.set_ylabel('Frequency')
# ax2.legend(loc='upper left')
ax2.grid(True)

# KDE plot
sns.kdeplot(df3[str(frameworkName[0])+str(AttackPercentage[0])+"_kde_cdf_m"], label='Malicious Advices', fill=True, ax=ax3, color='r', alpha=0.6)
sns.kdeplot(df6[str(frameworkName[0])+str(AttackPercentage[0])+"_kde_cdf_b"], label='Benign Advices', fill=True, ax=ax3, color='g', alpha=0.5)
ax3.set_title('KDE')
ax3.set_xlabel('Anomaly Score')
ax3.set_ylabel('Density')
# ax3.legend(loc='upper left')
ax3.grid(True)

# CDF plot
ax4.hist(df3[str(frameworkName[0])+str(AttackPercentage[0])+"_kde_cdf_m"], bins=50, density=True, cumulative=True, label='Malicious Advices', alpha=0.6, color='r')
ax4.hist(df6[str(frameworkName[0])+str(AttackPercentage[0])+"_kde_cdf_b"], bins=50, density=True, cumulative=True, label='Benign Advices', alpha=0.8, color='g')
ax4.set_title('CDF')
ax4.set_xlabel('Anomaly Score')
ax4.set_ylabel('Probability')
# ax4.legend(loc='upper left')
ax4.grid(True)

handles, labels = ax4.get_legend_handles_labels()
fig.legend(handles, labels, loc ='upper right', bbox_to_anchor=(0.65, 1.03), #0.25,-1,
         ncol=2, fancybox=True, shadow=True,
          borderpad=0.2,labelspacing=0.9, handlelength=0.9,columnspacing=0.8,handletextpad=0.3)

fig.text(0.15, 0.01, '(a)', ha='center', va='center', fontsize=15)
fig.text(0.40, 0.01, '(b)', ha='center', va='center', fontsize=15)
fig.text(0.65, 0.01, '(c)', ha='center', va='center', fontsize=15)
fig.text(0.89, 0.01, '(d)', ha='center', va='center', fontsize=15)


plt.tight_layout()

## Make sure path exists for this one line of code #################################
plt.savefig('./Anomaly_score_'+str(envType)+'.pdf',bbox_inches='tight') 
################################################################################

plt.show()


