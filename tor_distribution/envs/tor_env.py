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
import math
import matplotlib.pyplot as plt



class TorEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, cfg_file, net_file, route_file, vTypes_files, out_csv_name=None, sim_steps=10, use_gui=True, num_seconds=20000, max_depart_delay=100000, time_to_load_vehicles=0, delta_time=5, ):
        """
        initialization of the environment

        Observation:
            Type: Box(4)
            Num	Observation                Min            Max
            1	Cells CAV_CV #n           -Inf            Inf
            2	Celss pendingToCVehs #n   -Inf            Inf
            3	Celss LVsInToCZone #n     -Inf            Inf
            4	Cells Average Speed       -Inf            Inf
            5	Cells Veh Density         -Inf            Inf
        Actions:
            Type: Discrete(2)
            Num	Action
            1	Send ToC message Cell 1
            .
            10  Send ToC message Cell 10
            0	Not Send ToC message

        """

        super(TorEnv, self).__init__()
        self._cfg = cfg_file
        self.cells = None
        self.lanes_per_cell = None
        self._network = net_file
        self.last_reward = 0
        self.last_measurement = 0
        self._route = route_file
        self._vTypes = vTypes_files
        self.early = 0
        self.late = 0
        # self.sim_max_time=4833.5
        # self.sim_max_steps=48335
        self.sim_max_steps=sim_steps
        self.sim_max_time=self.sim_max_steps/10.0
        self.use_gui = use_gui
        self.x = []
        self.y = []
        self._sumo_binary = sumolib.checkBinary('sumo')

        seperator = ', '
        vTypesToString = seperator.join(self._vTypes)
        addFilesString = "scenario/additionalsOutput_trafficMix_0_trafficDemand_1_driverBehaviour_OS_seed_0.xml, "+ vTypesToString + ", scenario/shapes.add.xml, scenario/view.add.xml"
        sumo_args = ["-c", self._cfg,
                    "-a", addFilesString]

        traci.start([sumolib.checkBinary('sumo')] + sumo_args)

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(5,10), dtype=np.float32)
        self.action_space = spaces.Discrete(11)

        self.reward_range = (-float('inf'), float('inf'))
        self.run = 0
        self.myManager = TraciManager()
        self.density = {}
        self.occupancy = {}
        self.meanSpeed = {}
        self.waitingTime = {}
        self.trafficJams = {}
        self.keepAutonomy = {}
        self.plot = False
        self.delay = 0
        traci.close()


    def reset(self):
        """
        reset the enviromnent
        """
        if self.run != 0:
            self.myManager = TraciManager()
            traci.close()
            self.save_csv("outputs/data", self.run)


            # self.run = 0
        self.run += 1
        self.metrics = []
        self.last_measurement = 0
        self.early = 0
        self.late = 0
        self.x = []
        self.y = []
        self.dealy =0

        seperator = ', '
        vTypesToString = seperator.join(self._vTypes)
        addFilesString = "scenario/additionalsOutput_trafficMix_0_trafficDemand_1_driverBehaviour_OS_seed_0.xml, "+ vTypesToString + ", scenario/shapes.add.xml, scenario/view.add.xml"

        sumo_args = ["-c", self._cfg,
                     "--random",
                     "-r", self._route,
                     "-a", addFilesString]

        traci.start([self._sumo_binary] + sumo_args)
        return self._compute_observations()


    def _compute_observations(self):
        av =  self.myManager.getAVperCells()
        if not av:
            av = self.myManager.zerolistmaker(10)

        pend =  self.myManager.getPendperCells()
        if not pend:
            pend = self.myManager.zerolistmaker(10)

        lv =  self.myManager.getLVperCells()
        if not lv:
            lv = self.myManager.zerolistmaker(10)

        speeds = self.myManager.getSpeedPerCells()
        if not speeds:
            speeds = self.myManager.zerolistmaker(10)

        density = self.myManager.getDensityPerCells()
        if not density:
            density = self.myManager.zerolistmaker(10)

        return np.array([av[:10], pend[:10], lv[:10], speeds[:10], density[:10]], dtype=np.float32)


    def step(self, action):
        """
        execute environment step
        """

        # if self.delay<100:
        #    for i in range(100):
        #        self.delay +=1
        #        self._sumo_step()
        #
        # else:
        self._apply_actions(action)
        self._sumo_step()
        # observe new state and reward
        observation = self._compute_observations()
        # print("OBS " + str(observation[4]))
        reward = self._compute_rewards(action,observation)
        done =  self.myManager.sim_step() >= self.sim_max_time
        if(done):
            print(self.myManager.ToC_Per_Cell)
            if(self.plot):
                plt.plot(self.x, self.y)
                plt.xlabel('x - axis - Timestamp')
                plt.ylabel('y - axis - Mean Speed')
                plt.title('Line graph!')
                plt.show()
        info = self._compute_step_info(action)
        self.metrics.append(info)
        self.last_reward = reward

        return observation, reward, done, info


    def _apply_actions(self, action):
        """
        apply the actions in the environment
        More specifically store the activatedCell.
        """
        if (action != 0):
            self.myManager.activatedCell=action


    def _compute_rewards(self, action, observation):
        """
        We compute the rewards base on a specific function.
        There is also code for the plotting part.
        """
        # Store the sum of the Mean speed of the 2 lanes for plotting purposes
        lanes = self.myManager.getAreaLanes()
        ms = 0
        wt = 0
        for i in range(2):
            wt += self.myManager.getLaneWait(lanes[i])
            ms  += self.myManager.getLaneMeanSpeed(lanes[i])
        self.y.append(ms)

        # Calculate the reward
        # reward = self.reward_based_on_ToCs(action, observation)
        # reward = self.reward_based_on_Mean_Speed(action, observation)
        # reward = self.reward_based_on_Density(action, observation)
        reward = self.r(action, observation)
        # reward = self.reward_based_on_Travel_Time(action, observation)

        return reward


    def reward_based_on_ToCs(self, action, observation):
        """ Calculated reward based on the number of ToCs that we sent """
        reward = 0

        # punishment for the sum of forced ToCs
        reward = -(10*self.myManager.get_forced_ToCs())

        if(action != 0):
            reward += self.myManager.getDecidedToCs()*self.myManager.getCellInfluence(action)
        else:
            reward += 0

        return reward

    def reward_based_on_Mean_Speed(self, action, observation):
        """ Calculated reward based on the mean speed of the lanes """
        reward = 0

        # punishment for the sum of forced ToCs
        reward = -(50*self.myManager.get_forced_ToCs())

        lanes = self.myManager.getAreaLanes()
        ms = 0
        for i in range(2):
            ms  += self.myManager.getLaneMeanSpeed(lanes[i])

        if(action != 0):
            reward += self.myManager.getCellInfluence(action)*ms
        else:
            reward += 0.1 * ms

        return reward

    def reward_based_on_Density(self, action, observation):
        """ Calculated reward based on the densityPerCell """
        reward = 0
        densities = []
        densities = observation[4]

        # punishment for the sum of forced ToCs
        # reward = -(10*self.myManager.get_forced_ToCs())
        # if(action != 11):
        #     reward += -densities[action-1] + densities[action-1]*self.myManager.getCellInfluence(action)
        # else:
        #     reward += 0

        # different approach
        pun = self.myManager.get_forced_ToCs()
        if(pun!=0):
            reward  = -(pun)
        else:
            if(action != 0):
                reward = (1-densities[action-1])
            else:
                reward = sum(densities)/10

        return reward

    def r(self, action, observation):
        """ Calculated reward based on the ratio between
        the avg covered distance per cav/cv and TravelTime of the lanes """

        pun = self.myManager.get_forced_ToCs()
        if(sum(self.myManager.ToC_Per_Cell)!=0):
            cav_avg = self.myManager.cav_dist/sum(self.myManager.ToC_Per_Cell)
        else:
            cav_avg = 0
        tt=0
        lanes = self.myManager.getAreaLanes()
        for i in range(2):
            tt  += self.myManager.getLaneTravelTime(lanes[i])

        if(pun!=0):
            reward  = -(pun)
        else:
            if(action != 0):
                reward = cav_avg/tt
            else:
                reward = 0

        return reward

    def reward_based_on_Travel_Time(self, action, observation):
        """ Calculated reward based on the TravelTime of the lanes """
        reward = 0
        tt = 0
        final_reward = 0
        lanes = self.myManager.getAreaLanes()
        for i in range(2):
            tt  += self.myManager.getLaneTravelTime(lanes[i])

        pun = self.myManager.get_forced_ToCs()
        if(pun!=0):
            reward  = (pun*200)
        else:
            reward = tt

            # if(action != 11):
            #     reward = (1-self.myManager.getCellInfluence(action))*tt
            # else:
            #     reward = 0.9 * tt
        final_reward = self.last_measurement - reward
        self.last_measurement = reward
        return final_reward



    def _sumo_step(self):
        """
        execute simulation step
        """
        self.myManager.call_runner()
        self.x.append(self.myManager.getStep())


    def _compute_step_info(self,action):
        """
        get information at each simulation step
        """
        if(sum(self.myManager.ToC_Per_Cell)!=0):
            cav_avg = self.myManager.cav_dist/sum(self.myManager.ToC_Per_Cell)
        else:
            cav_avg = 0
        return {
            'step_time': self.myManager.sim_step(),
            'action': action,
            'avg_cav_dist' : cav_avg,
            'reward': self.last_reward,
            'last_measurement': self.last_measurement
            }


    def close(self):
        """
        close the simulation
        """
        traci.close()


    def render(self):
        """
        load gui for the simulation
        """
        self._sumo_binary = sumolib.checkBinary('sumo-gui')
        self.plot = True


    def save_csv(self, out_csv_name, run):
        """ save info to csv file """
        if out_csv_name is not None:
            df = pd.DataFrame(self.metrics)
            df.to_csv(out_csv_name + '_run{}'.format(run) + '.csv', index=False)
