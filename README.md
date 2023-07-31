# RAMPART: Reinforcing Autonomous Multi-agent Protection through Adversarial Resistance in Transportation

This is the codification used in the **ACM Journal on Autonomous Transportation Systems** (_Special Issue on Cybersecurity and Resiliency for Transportation Cyber-Physical Systems_) paper proposing RAMPART framework as a novel defense method against adversarial poisoning attacks in cooperative multiagent reinforcement learning (CMARL) algorithms in autonomous transportation. You are free to use all or part of the codes here presented for any purpose, provided that the paper is properly cited and the original authors properly credited. All the files here shared come with no warranties.


This project was built on Python 3.8. All the experiments are executed in the private payload delivery network (PPD) architecture, we included the version we used in the **Main/PP_environment** folder. For the graph generation code you will need to install Jupyter Notebook (http://jupyter.readthedocs.io/en/latest/install.html).

## Abstract
In the field of multi-agent autonomous transportation, such as automated payload delivery or highway on-ramp merging, agents routinely exchange knowledge to optimize their shared objective and adapt to environmental novelties through Cooperative Multi-Agent Reinforcement Learning (CMARL) algorithms. This knowledge exchange between agents allows these systems to operate efficiently and adapt to dynamic environments. However, this cooperative learning process is susceptible to adversarial poisoning attacks, as highlighted by contemporary research. Particularly, the poisoning attacks where malicious agents inject deceptive information camouflaged within the differential noise, a pivotal element for differential privacy (DP)-based CMARL algorithms, pose formidable challenges to identify and overcome. The consequences of not addressing this issue are far-reaching, potentially jeopardizing safety-critical operations and the integrity of data privacy in these applications. Existing research has strived to develop anomaly detection-based defense models to counteract conventional poisoning methods. Nonetheless, the recurring necessity for model offloading and retraining with labeled anomalous data undermines their practicality, considering the inherently dynamic nature of the safety-critical autonomous transportation applications. Further, it is imperative to maintain data privacy, ensure high performance, and adapt to environmental changes. Motivated by these challenges, this paper introduces a novel defense mechanism against stealthy adversarial poisoning attacks in the autonomous transportation domain, termed Reinforcing Autonomous Multi-agent Protection through Adversarial Resistance in Transportation (RAMPART). Leveraging a GAN model at each local node, RAMPART effectively filters out malicious advice in an unsupervised manner, whilst generating synthetic samples for each state-action pair to accommodate environmental uncertainties and eliminate the need for labeled training data. Our extensive experimental analysis, conducted in a Private Payload Delivery Network (PPDN) —a common application in the autonomous multi-agent transportation domain—demonstrates that \textbf{RAMPART successfully defends against a DP-exploited poisoning attack with a $30\%$ attack ratio, achieving an F1 score of $0.852$ and accuracy of $96.3\%$ in heavy-traffic environments}.


## Files
The folder **Main** contains our implementation of all algorithms and experiments

The folder **Main/PP_environment** contains the PPDN environment we used for experiments

Finally, the folder **ProcessedFiles** contains already processed files for graph printing and data visualization

## How to use <br />
First, install python 3.8 from https://www.python.org/downloads/release/python-380/<br />
Then open up your command terminal/prompt to run the following commands sequentially<br />
1. python RandomInit.py G N O E L Nw
2. python RAMPART_main.py G N O E L Nw Ap D S M Et gl ge gE gaD


where, <br />
G: Grid Height and Width (N x N)<br />
N: number of agents<br />
O: number of obstacles<br />
E: Total Episode<br />
L: number of times the code will run as a loop<br />
Nw: Neighbor weights [0,1]<br /> <br />

Ap: Attack Percentage [0,100]<br />
D: Display environment [on, off]<br />
S: Sleep (sec)<br />
M: Play mode [random, static]<br />
Et: Environment type [low, medium, heavy] <br />
gl: GAN loss threshold [0,1] <br />
ge: GAN training epoch <br />
gE: GAN training Episode <br />
gaD: GAN anomaly detection threshold [0,1] <br />

<br />
Example:<br />

```
python RandomInit.py 15 20 5 2000 10 0.90
python RAMPART_main.py 15 20 5 2000 10 0.90 20 on 2 random heavy 0.95 1000 100 0.8
```

<br /><br />
         
However, it might take a long time until all the experiments are completed. 
It may be of interest to run more than one algorithm at the same time if you have enough computing power. 
Also, note that, for each framework, if the AVs do not attain goal within (GridSize*100) steps in a particular episode, the episode and environment will be reset to the next. <br /><br />

The **file name** associated with any experiment is appended into a log file (RAMPART.txt) that resides inside "Main/OutputFile" directory.
The results (Steps to goal (SG), Rewards, Convergence) of any experiment are stored categorically by file name in "Main/SG", "Main/Reward", "Main/Convergence" respectively as a pickle file.
<br />
**Graph Generation and Reproduction**
1. Open processing.py file from "Main/" folder. Edit line 26-39 according to your experiments. Then run following command
	_python processing.py episode_num sg_gap reward_gap conv_gap disGap genGap nnGap atnGap env_type_
	where <br />
		episode_num = number of episode<br />
		sg_gap = plotting gap between SG values<br />
		reward_gap = plotting gap between Reward values<br />
  		conv_gap = plotting gap between Convergence values<br />
  		disGap = plotting gap between discriminator loss values <br />
    		genGap = plotting gap between generator loss values <br />
   		nnGap = plotting gap between no-attack neighbor (nn) (i.e., benign neighbors) anomaly score values <br />
     		atnGap = plotting gap between attack neighbor (atn) (i.e., malicious neighbors) anomaly score values <br />
    		env_type = environment type [Option: low/medium/heavy]<br />
Example: _```python processing.py 5000 500 500 20 1 1 1 1 low```_ <br /><br />

Your processed output will be stored inside the "Main/ProcessedOutput" folder in .csv format. Example output files are: ProcessedSG.csv, ProcessedReward.csv, ProcessedConvergence.csv, etc.<br /><br />
2. Then one-by-one run "Main/graph_SG.py", "Main/graph_reward.py", "Main/graph_convergence.py", etc. through below example steps.<br /><br />
	a. Open Main/graph_SG.py and edit line 49-56 as per your experiment and graph generation preferences<br />
	b. run _```python graph_SG.py episode_num gap env_type```_   (example: _```python graph_SG.py 5000 500 low```_)<br /><br />
	c. Open Main/graph_reward.py and edit line 50-57 as per your experiment and graph generation preferences<br />
	d. run _```python graph_reward.py episode_num gap env_type```_  (example: _```python graphGenerator_Reward.py 5000 500 low```_)<br /><br />
	e. Open Main/graph_convergence.py and edit line 52-59 as per your experiment and graph generation preferences<br />
	f. run _```python graph_convergence.py episode_num gap env_type```_   (example: _```python graph_convergence.py 5000 20 low```_)<br /><br />
 	g. Open Main/graph_dis_gen_loss.py and edit line 45-53 as per your experiment and graph generation preferences<br />
	h. run _```python graph_dis_gen_loss.py env_type```_   (example: _```python graph_dis_gen_loss.py low```_)<br /><br />
 	i. Open Main/graph_anomalyScore.py and edit line 36-44 as per your experiment and graph generation preferences<br />
	j. run _```python graph_anomalyScore.py env_type```_   (example: _```python graph_anomalyScore.py low```_)<br /><br />
	
	
Your output graphs will be stored in "./Main" folder <br /><br />

3. For convenience, we include a "ProcessedFiles" folder that is already populated by the results of our experiments. <br />
	Processed outputs are already in the "ProcessedFiles/ProcssedOutput" folder.<br /><br />
	Simply, run the following commands from **./ProcessedFiles folder** to see the graphs we have included in our paper<br/>
	```
	python graph_SG.py 2000 1 low
	python graph_reward.py 2000 1 low
	python graph_convergence.py 2000 1 low
	python graph_dis_gen_loss.py low
	python graph_anomalyScore.py low
 	```
	
	
## Contact
For questions about the codification or paper, <br />please send an email to mdtamjidh@nevada.unr.edu or aralab2018@gmail.com.
