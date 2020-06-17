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
import time
from datetime import datetime





class TorEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, cfg_file, net_file, route_file, vTypes_files, delay=100, out_csv_name=None, sim_steps=200, seed = 1024, trains=2, plot= False, use_gui=True, sim_example=False,   forced_toc_pun=1.0, data_path = ""):
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
        self.trains = trains
        self.plot= plot
        self.delay = delay
        self.use_gui = use_gui
        self.x = []
        self.y = []
        self.forcedT = 0
        self.forcedTocPun = forced_toc_pun
        self.tt = []
        self.ms = []
        self.start= None
        self._sumo_binary = sumolib.checkBinary('sumo')
        self.data_path = data_path
        self.sim_example = sim_example
        self.cells_number = 14

        seperator = ', '
        vTypesToString = seperator.join(self._vTypes)
        addFilesString = "scenario/additionalsOutput_trafficMix_0_trafficDemand_1_driverBehaviour_OS_seed_0.xml, "+ vTypesToString + ", scenario/shapes.add.xml, scenario/view.add.xml"
        sumo_args = ["-c", self._cfg,
                    "-a", addFilesString]

        traci.start([sumolib.checkBinary('sumo')] + sumo_args)

        self.observation_space = spaces.Box(-np.inf, np.inf,
                                            shape=(5, self.cells_number), dtype=np.float32)
        self.action_space = spaces.Discrete(self.cells_number+1-2)

        self.reward_range = (-float('inf'), float('inf'))
        self.run = 0
        self.myManager = TraciManager(self.cells_number)
        self.density = {}
        self.occupancy = {}
        self.meanSpeed = {}
        self.waitingTime = {}
        self.trafficJams = {}
        self.keepAutonomy = {}
        self.plot = False
        self.delay = 0
        self.seed = seed
        traci.close()


    def reset(self):
        """
        reset the enviromnent
        """
        self.start = time.time()
        if self.run != 0:
            self.myManager = TraciManager(self.cells_number)
            traci.close()
            # self.save_csv(self.data_path, self.run)



        self.run += 1
        self.metrics = []
        self.last_measurement = 0
        self.early = 0
        self.late = 0
        self.x = []
        self.y = []
        self.forcedT = 0
        self.tt = []
        self.ms = []
        self.dealy =0


        seperator = ', '
        vTypesToString = seperator.join(self._vTypes)
        addFilesString = "scenario/additionalsOutput_trafficMix_0_trafficDemand_1_driverBehaviour_OS_seed_0.xml, "+ vTypesToString + ", scenario/shapes.add.xml, scenario/view.add.xml"
        if(self.sim_example):
            trip_path = self.data_path + "/tripinfo_" + str(self.run) + ".xml"
            print(self.seed+self.run)
            sumo_args = ["-c", self._cfg,
                        #  "--random",
                        "--seed", str(self.seed+self.run),
                         # "--tripinfo-output.write-unfinished", "True",
                         "--tripinfo-output",  trip_path,
                         "-r", self._route,
                         "-a", addFilesString]
        else:
            sumo_args = ["-c", self._cfg,
                         "--random",
                         "-r", self._route,
                         "-a", addFilesString]



        traci.start([self._sumo_binary] + sumo_args)
        for i in range(self.dealy):
            self._sumo_step()
        return self._compute_observations()


    def _compute_observations(self):
        tt=0
        ms = 0
        lanes = self.myManager.getAreaLanes()
        for i in range(2):
            tt  += self.myManager.getLaneTravelTime(lanes[i])
            ms  += self.myManager.getLaneMeanSpeed(lanes[i])
        self.tt.append(tt)
        self.ms.append(ms)
        self.x.append(self.myManager.getStep())

        av =  self.myManager.getAVperCells()
        if not av:
            av = self.myManager.zerolistmaker(self.cells_number)

        pend =  self.myManager.getPendperCells()
        if not pend:
            pend = self.myManager.zerolistmaker(self.cells_number)

        lv =  self.myManager.getLVperCells()
        if not lv:
            lv = self.myManager.zerolistmaker(self.cells_number)

        speeds = self.myManager.getSpeedPerCells()
        if not speeds:
            speeds = self.myManager.zerolistmaker(self.cells_number)

        density = self.myManager.getDensityPerCells()
        if not density:
            density = self.myManager.zerolistmaker(self.cells_number)

        return np.array([av[:self.cells_number], pend[:self.cells_number], lv[:self.cells_number], speeds[:self.cells_number], density[:self.cells_number]], dtype=np.float32)


    def step(self, action):
        """
        execute environment step
        """

        self._apply_actions(action)
        self._sumo_step()
        # observe new state and reward
        observation = self._compute_observations()
        # print("OBS " + str(observation[4]))
        reward = self._compute_rewards(action,observation)
        done =  self.myManager.sim_step() >= self.sim_max_time
        if(done):
            print()
            elapsed_time = time.time()- self.start
            print("Duration of the phase =", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
            if(self.sim_example==False):
                print("Train Episode " + str(self.run) +  " of " + str(self.trains) + " finished after " + str(self.sim_max_steps) + " timesteps")
            else:
                print("Simulation example finished after " + str(self.sim_max_steps) + " timesteps")
            print("Number of ToC per cell")
            print(self.myManager.ToC_Per_Cell)

            print("Total number of ToC messages: " + str(sum(self.myManager.ToC_Per_Cell)))
            print("Number of forced ToC messages: " + str(self.myManager.forcedToCs))
            print("Average covered distance of CAV_CV vehs: " + str(self.myManager.cav_dist/sum(self.myManager.ToC_Per_Cell)))
            print("Average Mean Speed in simulation for both of the lanes: " + str(sum(self.ms)/len(self.ms)))
            print("Average Travel Time in simulation for both of the lanes: " + str(sum(self.tt)/len(self.tt)))
            print()
            self.save_csv(self.data_path, self.run)

            # if(self.plot):
            #     self.myplot(self.x,self.tt,"Timestamp", "TravelTime")

        info = self._compute_step_info(action,observation)
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

        # Calculate the reward
        # reward = self.reward_based_on_ToCs(action, observation)
        # reward = self.reward_based_on_Mean_Speed(action, observation)
        # reward = self.reward_based_on_Density(action, observation)
        # reward = self.r(action, observation)
        # reward = self.reward_based_on_Travel_Time(action, observation)
        # reward = self.reward_based_on_Distribution(action, observation)
        reward = self.reward_based_on_Distribution_Speed(action, observation)


        return reward


    def reward_based_on_Distribution(self, action, observation):
        """ Calculated reward based on the number of ToCs that we sent """
        reward = 0

        # punishment for the sum of forced ToCs
        tt=0
        ms = 0
        lanes = self.myManager.getAreaLanes()
        wt = 0
        for i in range(2):
            wt += self.myManager.getLaneWait(lanes[i])
            tt  += self.myManager.getLaneTravelTime(lanes[i])
            ms  += self.myManager.getLaneMeanSpeed(lanes[i])
        pun = 0
        pun = self.forcedTocPun*self.myManager.get_forced_ToCs()
        self.forcedT = self.myManager.get_forced_ToCs()
        wt_pun=0
        if(wt!=0):
            wt_pun=10*wt

        if(action != 0):
            if(observation[1][action-1]<4 and pun ==0 and  wt_pun==0):
                reward = 10 + self.myManager.getCellInfluence(action)
            else:
                reward = -(observation[1][action-1] + pun+wt_pun) + self.myManager.getCellInfluence(action)
        else:
            reward = 0 - pun - wt_pun

        return reward

    def reward_based_on_Distribution_Speed(self, action, observation):
        """ Calculated reward based on the number of ToCs that we sent """
        reward = 0

        # punishment for the sum of forced ToCs
        tt = 0
        ms = 0
        lanes = self.myManager.getAreaLanes()
        wt = 0

        for i in range(2):
            wt += self.myManager.getLaneWait(lanes[i])
            tt += self.myManager.getLaneTravelTime(lanes[i])
            ms += self.myManager.getLaneMeanSpeed(lanes[i])
        pun = 0
        # pun = self.forcedTocPun*self.myManager.get_forced_ToCs()
        self.forcedT = self.myManager.get_forced_ToCs()
        wt_pun = 0
        speed_pun = 0


        # for i in range(self.cells_number):
        #     if(i>=2):
        #         if(observation[3][i] <= 20 and (observation[0][i]+observation[1][i]+observation[2][i]) >0):
        #             speed_pun+=10
        if(observation[3][(action-1)+2] <= 25 and (observation[0][(action-1)+2]+observation[1][(action-1)+2]+observation[2][(action-1)+2]) > 0):
            speed_pun = observation[3][(action-1)+2]

        if(wt != 0):
            # print("WT ",wt)
            # print("speeds ", observation[3])
            wt_pun = 10*wt
        if(self.cells_number ==10):
            limit = 4
        else:
            limit = 3

        if(action != 0):
            if(observation[1][action-1] < limit and pun == 0 and wt_pun == 0 and speed_pun ==0):
                reward = 10 #+ self.myManager.getCellInfluence(action) 
            else:
                reward = -(observation[1][action-1] + pun +wt_pun + speed_pun) #+ self.myManager.getCellInfluence(action)
        else:
            reward = 0 - pun - wt_pun - speed_pun

        return reward


    def _sumo_step(self):
        """
        execute simulation step
        """
        self.myManager.call_runner()


    def _compute_step_info(self,action,observation):
        """
        get information at each simulation step
        """
        if(sum(self.myManager.ToC_Per_Cell)!=0):
            cav_avg = self.myManager.cav_dist/sum(self.myManager.ToC_Per_Cell)
        else:
            cav_avg = 0

        return {
            'step_time': float(self.myManager.sim_step()),
            'action': action,
            'avg_cav_dist' : cav_avg,
            'reward': self.last_reward,
            'meanSpeed':  self.ms[-1],
            'TravelTime': self.tt[-1],
            'ForcedT': self.forcedT,
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
        self.sim_example = True
        self.plot = True


    def save_csv(self, out_csv_name, run):
        """ save info to csv file """
        if out_csv_name is not None:
            df = pd.DataFrame(self.metrics)
            df.to_csv(out_csv_name + '/data_run{}'.format(run) + '.csv', index=False)


    def myplot(self, x, y, xName, yName):
        """ plot stuff """
        plt.plot(x, y)
        plt.xlabel('x - axis - ' +xName)
        plt.ylabel('y - axis - ' +yName)
        plt.title('Line graph!')
        plt.show()
        # plt.plot(self.x, self.tt)
        # plt.xlabel('x - axis - Timestamp')
        # plt.ylabel('y - axis - Travel Time')
        # plt.title('Line graph!')
        # plt.show()


    # Previous reward functions   

    # def reward_based_on_ToCs(self, action, observation):
    #     """ Calculated reward based on the number of ToCs that we sent """
    #     reward = 0

    #     # punishment for the sum of forced ToCs
    #     reward = -(10*self.myManager.get_forced_ToCs())

    #     if(action != 0):
    #         reward += self.myManager.getDecidedToCs()*self.myManager.getCellInfluence(action)
    #     else:
    #         reward += 0

    #     return reward


    # def reward_based_on_Mean_Speed(self, action, observation):
    #     """ Calculated reward based on the mean speed of the lanes """
    #     reward = 0

    #     # punishment for the sum of forced ToCs
    #     reward = -(50*self.myManager.get_forced_ToCs())

    #     lanes = self.myManager.getAreaLanes()
    #     ms = 0
    #     for i in range(2):
    #         ms  += self.myManager.getLaneMeanSpeed(lanes[i])

    #     if(action != 0):
    #         reward += self.myManager.getCellInfluence(action)*ms
    #     else:
    #         reward += 0.1 * ms

    #     return reward

    # def reward_based_on_Density(self, action, observation):
    #     """ Calculated reward based on the densityPerCell """
    #     reward = 0
    #     densities = []
    #     densities = observation[4]

    #     # punishment for the sum of forced ToCs
    #     # reward = -(10*self.myManager.get_forced_ToCs())
    #     # if(action != 11):
    #     #     reward += -densities[action-1] + densities[action-1]*self.myManager.getCellInfluence(action)
    #     # else:
    #     #     reward += 0

    #     # different approach
    #     pun = self.myManager.get_forced_ToCs()
    #     if(pun!=0):
    #         reward  = -(pun)
    #     else:
    #         if(action != 0):
    #             reward = (1-densities[action-1])
    #         else:
    #             reward = sum(densities)/10

    #     return reward

    # def r(self, action, observation):
    #     """ Calculated reward based on the ratio between
    #     the avg covered distance per cav/cv and TravelTime of the lanes """
    #     msC = observation[3]

    #     pun = self.myManager.get_forced_ToCs()
    #     if(sum(self.myManager.ToC_Per_Cell)!=0):
    #         cav_avg = self.myManager.cav_dist/sum(self.myManager.ToC_Per_Cell)
    #     else:
    #         cav_avg = 0
    #     tt=0
    #     ms = 0
    #     lanes = self.myManager.getAreaLanes()
    #     for i in range(2):
    #         tt  += self.myManager.getLaneTravelTime(lanes[i])
    #         ms  += self.myManager.getLaneMeanSpeed(lanes[i])

    #     df = 1 - ((tt/150) + (60/ms))
    #     # dfs = 1 - (60/ms)

    #     if(pun!=0):
    #         reward  = -(pun)

    #     else:
    #         reward = cav_avg * df

    #     return reward

    # def reward_based_on_Travel_Time(self, action, observation):
    #     """ Calculated reward based on the TravelTime of the lanes """
    #     reward = 0
    #     tt = 0
    #     final_reward = 0
    #     lanes = self.myManager.getAreaLanes()
    #     for i in range(2):
    #         tt  += self.myManager.getLaneTravelTime(lanes[i])

    #     pun = self.myManager.get_forced_ToCs()
    #     self.forcedT = pun

    #     # if(pun !=0):
    #     #     reward  = tt*pun
    #     # elif(tt >300):
    #     #     reward =1000
    #     # else:
    #     #     if(action != 11):
    #     #         reward = (1-self.myManager.getCellInfluence(action))*tt
    #     #     else:
    #     #         reward = tt/2

    #         # if(action != 11):
    #         #     reward = (1-self.myManager.getCellInfluence(action))*tt
    #         # else:
    #         #     reward = 0.9 * tt
    #     if(sum(observation[0])==0):
    #         tt=0
    #     final_reward = self.last_measurement - tt
    #     self.last_measurement = tt
    #     return final_reward
    #     # return -reward


