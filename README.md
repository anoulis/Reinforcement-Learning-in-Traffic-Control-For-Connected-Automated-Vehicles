# Multi Agent Approach for ToR Distribution 

This project aims to train multi agents with reinforcement learning to distribute take-over-requests (ToR) for automated connected vehicles.

---
## Installation Requirements
1.) Download the code of the SingleAgent branch

    git clone https://gitlab.dlr.de/ml-with-flow/tor-distribution -b SingleAgent
    cd tor-distibution


2.) Install Anacond from https://www.anaconda.com/download

3.) Install SUMO (SUMO version used 1.6) https://sumo.dlr.de/docs/Installing.html

Note: Make sure to setup your SUMO_HOME variable.


### Set up conda env

```
conda env create -f environment.yml
conda activate ma-tor
pip install ray[rllib]
```


### Troubleshooting

For any cross environmet / platforms problem
* If you have problem in creation of the environment for installing the specific version of any of the dependencies, install the latest available.
* For example: You cannot install the tensorflow-gpu=2.2.0. Install the latest available of the tensorflow-gpu=2


### Overview of the files

```
* tor-distribution/main_ma.py : The main file for executing trainings and simulations
* tor-distribution/rollout.py : The customized version of the rollout file of rllib. Needed for executing the simulations.
* tor-distribution/myDQNTFPolicy.py: File for inserting your custom DQN policy (not used now).
* tor-distribution/callbacks.py : File for inserting callbacks during training (not needed for now).
* tor-distribution/tor_distribution/envs/tor_env.py : reinforcement learning environment
* tor-distribution/tor_distribution/envs/traci_manager.py : communication with TraCI API
* tor-distribution/utils/tools.py : creating the comparison charts
* tor-distribution/final_models/zips : folder with already pretrained Single Agent model
* tor-distribution/final_models/sims : folder with the already acquired simulations files from the pretrained Single Agent models
```

### General Remarks for the Multi Agent Approach

* The dedault location is the directory of the tor-distribution folder.
* The default number for trainings is 30.
* The default number of simulations is 10.
* Number of cells supported: 14.
* Number of agents supported: 2,3,6.
* The user can set the nuber of agents in main_ma file.
* The default reward function is the last one that had the best results.
* There are also other available for experimentation. The choice is happening in the _compute_rewards function.
* In the training process, we save important information in the /outputs/ray_results/. 
* Also in same location, there are the checkpoints for the model (save after every iteration) in seperate folders.
* Before running simulations, the user need to put the location of the model and last checkpoint, (s)he wants to evaluate.
* In the simulations process, we save important information(csv and tripinfo files) in a folder with same name with the model in location outputs/simulations. 
* By appending the simulations folder location at the beggining of utils/tools.py, the user can compare the results with other save ones.
* The command "python utils/tools.py" runs the comparions. The results are in the location outputs/simulations/comparisons.
* It uses the default value for simulations (10) , for other number should modify the tool file.


### Run the training with default arguments

```
python main_ma.py -mode train
```

* The results will be saved in a folder : "outputs/ray_results/DQN__trainedDate/
* The last checkpoint will be saved in folder : "outputs/ray_results/DQN__trainedDate_randomCode/checkpoint_30

### Run the training with all the available custom arguments

```
python main_ma.py -mode train -trains 15 -sim_steps 10000

```
This will do the following:
* will train the model for 15 times.
* The simulation will last for 10000 steps.
* The results will be saved in a folder : "outputs/ray_results/DQN__trainedDate/
* The last checkpoint will be saved in folder : "outputs/ray_results/DQN__trainedDate_randomCode/checkpoint_15


### Run the simulation with default arguments

```
python main_ma.py - mode eval
```
** Before running the evalution, the user need to put the location of the model and last checkpoint, (s)he wants to evaluate.
* In main_ma.py, line 105, value "eval" should be like : eval_path = "DQN__trainedDate_randomCode/"

This will do the following:
* The trained model will run 10 simulation rounds.
* The simulation will last for 48335 steps.
* The results will be saved in a folder : "outputs/simulations/DQN__trainedDate_simulatedDate/"

### Run the simulation with all the available custom arguments

```
python main_ma.py - mode eval -simulations 15 -sim_steps 10000
```
** Before running the evalution, the user need to put the location of the model and last checkpoint, (s)he wants to evaluate.
* In main_ma.py, line 105, value "eval" should be like : eval_path = "DQN__trainedDate_randomCode/


This will do the following:
* The trained model will run 15 simulation rounds.
* The simulation will last for 10000 steps.
* The results will be saved in a folder : "outputs/simulations/DQN__trainedDate_simulatedDate/

### Run tensorboard
* To use tensorboard, the trained files should be moved to default ray location in home directory of user (/ray_results).
* After that use the following command for all or one specific experinent

```
tensorboard --logdir=~/ray_results/
******************or********************
tensorboard --logdir=~/ray_results/my_experiment
```


