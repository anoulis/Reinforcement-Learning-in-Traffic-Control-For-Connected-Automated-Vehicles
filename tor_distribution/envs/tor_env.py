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


class TorEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, cfg_file, net_file, route_file, vTypes_files, out_csv_name=None, use_gui=True, num_seconds=20000, max_depart_delay=100000, time_to_load_vehicles=0, delta_time=5, ):
        super(TorEnv, self).__init__()
        self._cfg = cfg_file
        self.cells = 2
        self.lanes_per_cell = 1
        self._network = net_file
        self.last_reward = 1
        self._route = route_file
        self._vTypes = vTypes_files
        self.sim_max_time=100000
        self.use_gui = use_gui
        if self.use_gui:
            self._sumo_binary = sumolib.checkBinary('sumo-gui')
        else:
            self._sumo_binary = sumolib.checkBinary('sumo')
        seperator = ', '
        vTypesToString = seperator.join(self._vTypes)
        addFilesString = "scenario/additionalsOutput_trafficMix_0_trafficDemand_1_driverBehaviour_OS_seed_0.xml, "+ vTypesToString + ", scenario/shapes.add.xml, scenario/view.add.xml"
        sumo_args = ["-c", self._cfg,
                    "-a", addFilesString]

        traci.start([sumolib.checkBinary('sumo')] + sumo_args)
        # self.observation_space = spaces.Box(low=np.zeros(2 + 1 + 2*2), high=np.ones(2+ 1 + 2*2))
        # self.discrete_observation_space = spaces.Tuple((
        #     spaces.Discrete(self.num_green_phases),                         # Green Phase
        #     spaces.Discrete(self.max_green//self.delta_time),               # Elapsed time of phase
        #     *(spaces.Discrete(10) for _ in range(2*self.lanes_per_ts))      # Density and stopped-density for each lane
        # ))
        # self.observation_space = spaces.Tuple((
        #     spaces.Box(0, max_wealth, shape=[1], dtype=np.float32),  # current wealth
        #     spaces.Discrete(max_rounds+1),  # rounds elapsed
        #     spaces.Discrete(max_rounds+1),  # wins
        #     spaces.Discrete(max_rounds+1),  # losses
        #     spaces.Box(0, max_wealth, [1], dtype=np.float32)))  # maximum observed wealth
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, dtype=np.float32, shape=(self.cells, 3, 1))
        self.action_space = spaces.Discrete(2)
        self.reward_range = (-float('inf'), float('inf'))
        self.run = 0
        self.metrics = []

        traci.close()

    def reset(self):
        if self.run != 0:
            traci.close()
            self.save_csv(self.out_csv_name, self.run)
        self.run += 1
        self.metrics = []
        seperator = ', '
        vTypesToString = seperator.join(self._vTypes)
        addFilesString = "scenario/additionalsOutput_trafficMix_0_trafficDemand_1_driverBehaviour_OS_seed_0.xml, "+ vTypesToString + ", scenario/shapes.add.xml, scenario/view.add.xml"

        sumo_args = ["-c", self._cfg,
                     "-r", self._route,
                     "-a", addFilesString]
        if self.use_gui:
            self._sumo_binary = sumolib.checkBinary('sumo-gui')

        traci.start([self._sumo_binary] + sumo_args)
        self.vehicles = dict()
        return self._compute_observations()



    def _compute_observations(self):
        """
        Return the current observation for each traffic signal
        """
        # observations = {}
        # for cell in self.cell_ids:
        num_av = 0
        density = 2 #get_lanes_density()
        queue = 3 #get_lanes_queue()
        observations = num_av + density + queue
        observations = np.ones((self.cells, 3, 1))
        # return (np.array([float(self.wealth)]), self.rounds_elapsed, self.wins,self.losses, np.array([float(self.max_ever_wealth)]))
        return observations

    @property
    def sim_step(self):
        """
        Return current simulation second on SUMO
        """
        return traci.simulation.getTime()

    def step(self, action):
        self._apply_actions(action)

        self._sumo_step()

        # observe new state and reward
        observation = self._compute_observations()
        reward = self._compute_rewards()
        done =  self.sim_step > self.sim_max_time
        info = self._compute_step_info()
        self.metrics.append(info)
        self.last_reward = reward

        return observation, reward, done, {}

    def _apply_actions(self, actions):

        # self.send_mess(actions)
        pass

    def _compute_rewards(self):
        return 1

    def _sumo_step(self):
        traci.simulationStep()

    def _compute_step_info(self):
        return {
            'step_time': self.sim_step,
            'reward': self.last_reward,
            'total_trans_msgs': 1
        }

    def close(self):
        traci.close()


    #
    #
    # def step(self, action):
    #     pass
    # def reset(self):
    #     pass
    # def render(self, mode='human'):
    #     pass
    # def close(self):
    #     pass
