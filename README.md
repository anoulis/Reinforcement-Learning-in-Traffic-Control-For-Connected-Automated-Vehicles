# TOR-distribution

This project aims to train an agent with reinforcement learning to distribute take-over-requests (TOR) for automated connected vehicles.

---
## Installing TOR-distribution and SUMO
1.) Download the code.

    git clone https://gitlab.dlr.de/ml-with-flow/tor-distribution
    cd tor-distibution


2.) Create an environment using https://www.anaconda.com/download

3.) Install SUMO https://sumo.dlr.de/docs/Installing.html
Note: Make sure to setup your SUMO_HOME variable.

test SUMO via ...



### Set up conda env

```
conda env create -f environment.yml
conda activate ma-tor
pip install ray[rllib]
```

### Run the experiment

```
python experiments/simple.py

```
