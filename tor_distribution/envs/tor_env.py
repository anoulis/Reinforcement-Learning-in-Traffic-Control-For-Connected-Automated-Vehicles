import gym
from gym import error, spaces, utils
from gym.utils import seeding

import os
import sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci
import sumolib
import traci.constants as tc
import numpy as np
import pandas as pd
from .traci_manager import TraciManager
import random


class TorEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, cfg_file, net_file, route_file, vTypes_files, out_csv_name=None, sim_steps=10, use_gui=True, num_seconds=20000, max_depart_delay=100000, time_to_load_vehicles=0, delta_time=5, ):
        """
        initialization of the environment

        Observation:
            Type: Box(4)
            Num	Observation               Min             Max
            0	Density lane 1           -Inf            Inf
            1	Density lane 2           -Inf            Inf
            2	More                     -Inf            Inf
            3	Another one              -Inf            Inf
        Actions:
            Type: Discrete(2)
            Num	Action
            0	Not Send ToC message
            1	Send ToC message
        """

        super(TorEnv, self).__init__()
        self._cfg = cfg_file
        self.cells = None
        self.lanes_per_cell = None
        self._network = net_file
        self.last_reward = 0
        self.last_mesauremnt = 0
        self._route = route_file
        self._vTypes = vTypes_files
        self.early = 0
        self.late = 0
        # self.sim_max_time=4833.5
        # self.sim_max_steps=48335
        self.sim_max_steps=sim_steps
        self.sim_max_time=self.sim_max_steps/10.0
        self.use_gui = use_gui
        # if self.use_gui:
        #     self._sumo_binary = sumolib.checkBinary('sumo-gui')
        # else:
        #     self._sumo_binary = sumolib.checkBinary('sumo')

        self._sumo_binary = sumolib.checkBinary('sumo')

        seperator = ', '
        vTypesToString = seperator.join(self._vTypes)
        addFilesString = "scenario/additionalsOutput_trafficMix_0_trafficDemand_1_driverBehaviour_OS_seed_0.xml, "+ vTypesToString + ", scenario/shapes.add.xml, scenario/view.add.xml"
        sumo_args = ["-c", self._cfg,
                    "-a", addFilesString]

        traci.start([sumolib.checkBinary('sumo')] + sumo_args)
        high = np.array([np.finfo(np.float32).max,
                        np.finfo(np.float32).max],
                        dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.reward_range = (-float('inf'), float('inf'))
        self.run = 0
        self.metrics = []
        self.myManager = TraciManager()
        traci.close()


    def reset(self):
        """
        reset the enviromnent
        """
        if self.run != 0:
            self.myManager = TraciManager()
            traci.close()
            self.run = 0
            # self.save_csv(self.out_csv_name, self.run)
        # self.run += 1
        self.metrics = []
        self.last_mesauremnt = 0
        self.early = 0
        self.late = 0

        seperator = ', '
        vTypesToString = seperator.join(self._vTypes)
        addFilesString = "scenario/additionalsOutput_trafficMix_0_trafficDemand_1_driverBehaviour_OS_seed_0.xml, "+ vTypesToString + ", scenario/shapes.add.xml, scenario/view.add.xml"

        sumo_args = ["-c", self._cfg,
                     "-r", self._route,
                     "-a", addFilesString]
        # if self.use_gui:
        #     self._sumo_binary = sumolib.checkBinary('sumo-gui')


        traci.start([self._sumo_binary] + sumo_args)
        return self._compute_observations()

    def _compute_observations(self):
        """
        Return the current observation values
        """
        # Temp mode
        observations = []
        num_av = self.myManager.getVehNum()
        density= []
        dets = self.myManager.getAreaDet()
        lanes = self.myManager.getAreaLanes()
        for i in range(2):
            density.append(self.myManager.getLaneVehNum(dets[i]) / traci.lane.getLength(lanes[i]) / 1000.)
        observations = np.array([density[0],density[1]],dtype=np.float32)

        return observations


    def step(self, action):
        """
        execute environment step
        """
        self._sumo_step(self._apply_actions(action))
        # observe new state and reward
        observation = self._compute_observations()
        reward = self._compute_rewards()
        done =  self.myManager.sim_step() >= self.sim_max_time
        info = self._compute_step_info()
        self.metrics.append(info)
        self.last_reward = reward

        return observation, reward, done, info

    def _apply_actions(self, action):
        """
        apply the actions in the environment
        """
        if (action == 1):
            return True
        else:
            return False

    def _compute_rewards(self):
        """
        compute the rewards for every action

        4 for evert early ToC message
        5 for every CAV/CV not get message before the last 500m of the zone
        """
        punishments =  self.myManager.get_late_punishment() + self.myManager.get_early_punishment()
        reward =  self.last_mesauremnt - punishments
        self.last_mesauremnt = punishments

        return reward

    def _sumo_step(self, flag):
        """
        execute simulation step
        """
        self.run += 1
        pun = self.myManager.call_runner(self.run,flag)
        return pun

    def _compute_step_info(self):
        """
        get information at each simulation step
        """
        return {
            'step_time': self.myManager.sim_step(),
            'reward': self.last_reward,
            'total_trans_msgs': 1,
            'total_trans_msgs': self.early,
            'total_trans_msgs': self.late
        }

    def close(self):
        """
        close the simulation
        """
        self.run = 0
        traci.close()

    def render(self):
        """
        load gui for the simulation
        """
        self._sumo_binary = sumolib.checkBinary('sumo-gui')
