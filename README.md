# Single Agent Approach for ToR Distribution 

This project aims to train an agent with reinforcement learning to distribute take-over-requests (ToR) for automated connected vehicles.

---
## Installation Requirements
1.) Download the code of the SingleAgent branch

    git clone https://gitlab.dlr.de/ml-with-flow/tor-distribution -b SingleAgent
    cd tor-distibution


2.) Install Anacond from https://www.anaconda.com/download

3.) Install SUMO (SUMO version used 1.6) https://sumo.dlr.de/docs/Installing.html
Note: Make sure to setup your SUMO_HOME variable.


### Set up the conda env and install gym env

```
conda env create -f environment.yml
pip install -e .
```

### Overview of the files

```
* experiments/server/training.py : file for executing trainings
* experiments/server/simulation.py : file for executing simulations
* tor-distribution/tor_distribution/envs/tor_env.py : reinforcement learning environment
* tor-distribution/tor_distribution/envs/traci_manager.py : communication with TraCI API
* tor-distribution/utils/tools.py : creating the comparison charts
* tor-distribution/final_models/zips : folder with already pretrained model
* tor-distribution/final_models/sims : folder with the already acquired simulations files from the pretrained models
```

### Run the training file with default arguments

```
python experiments/server/training.py

```

### Run the training file with all the available custom arguments

```
python experiments/server/training.py -trains 15 -sim_steps 10000 -zip sample.zip -delay 100 -pun 1.5

This will do the following:
* will train the model for 15 times.
* The simulation will last for 10000 steps.
* The saved model will be stored in the sample.zip.
* The delay in the simulation (time where the simulation will run but the Agent will be inactive) will be 100 simulation steps.
* The pun value is targeted to reward functions of the phase 2 that support it (is float number).

```


### Run the simulation file with default arguments

```
python experiments/server/simulation.py

```

### Run the training file with all the available custom arguments

```
python experiments/server/simulation.py -simulations 15 -sim_steps 10000 -zip sample.zip -delay 100 -pun 1.5

This will do the following:
* The trained model will run 15 simulation rounds.
* The simulation will last for 10000 steps.
* The saved model that will be restored for the simulations.
* The delay in the simulation (time where the simulation will run but the Agent will be inactive) will be 100 simulation steps.
* The pun value is targeted to reward functions of the phase 2 that support it (is float number).

```

### General Remarks for the Single Agent Approach

```
This will do the following:
* The default reward function is the last one that had the best results.
* There are also others available for experimentation. They can be selected from the _compute_rewards function.
Before run them, the user should pay attention to Notes at the beginning of each reward function, in order to adjuct observation and action space.
* In the training process, we save important information in the folder with same name with the zip file in location outputs/trainings. 
Also in same location, in log folder we save the checkpoints for the model (save after every iteration).
* In the simulations process, we save important information(csv and tripinfo files) in the folder with same name with the zip file in location outputs/simulations. 
* By appendins the simulations folder location at the beggining of utils/tools.py, the user can compare the results with other save ones.
* The command "python utils/tools.py" runs the comparions. The results are in the location outputs/simulations/comparisons.

```

