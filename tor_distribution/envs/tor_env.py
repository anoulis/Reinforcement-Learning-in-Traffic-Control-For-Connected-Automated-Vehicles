from ray.rllib.env.multi_agent_env import MultiAgentEnv
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


class TorEnv(MultiAgentEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, cfg_file=str("scenario/sumo.cfg"), net_file="scenario/UC5_1.net.xml", route_file='scenario/routes_trafficMix_0_trafficDemand_1_driverBehaviour_OS_seed_0.xml', vTypes_files=['scenario/vTypesCAVToC_OS.add.xml', 'scenario/vTypesCVToC_OS.add.xml', 'scenario/vTypesLV_OS.add.xml'], delay=0, out_csv_name=None, sim_steps=200, seed=1024, trains=2, plot=False, use_gui=True, sim_example=False,   forced_toc_pun=1.0, data_path="/home/anoulis/workspace/tor-distribution/outputs/trainings/", agents=2):
        """
        initialization of the environment

        Observation:
            Type: Box(4)
            Num	Observation                Min            Max
            1	Cells CAV_CV #n           -Inf            Inf
            2	Cells Average Speed       -Inf            Inf
            3	Celss pendingToCVehs #n   -Inf            Inf
            4	Celss SendedToCs     #n   -Inf            Inf
        Actions:
            Type: Discrete(2)
            Num	Action
            1	Send ToC message Cell 1
            .
            10  Send ToC message Cell 10
        """


        # self.rsu_ids = [0,1]
        # self.rsuActivatedCells = [0, 0]

        self._cfg = cfg_file
        self._network = net_file
        self.last_reward = 0
        self._route = route_file
        self._vTypes = vTypes_files
        self.total_reward = 0
        self.acted_times = 0
        # self.sim_max_time=4833.5
        # self.sim_max_steps=48335
        self.delay = delay
        self.sim_max_steps=sim_steps
        self.sim_max_time=self.sim_max_steps/10.0
        self.plot= plot
        self.use_gui = use_gui
        self.x = []
        self.forcedT = 0
        self.forcedTocPun = forced_toc_pun
        self.tt = []
        self.ms = []
        self.start= None
        self.trains = trains
        self._sumo_binary = sumolib.checkBinary('sumo')
        self.data_path = data_path
        self.sim_example = sim_example
        self.cells_number = 14
        self.agents = agents
        self.cellsPerAgent =int((self.cells_number-2)/self.agents)

        seperator = ', '
        vTypesToString = seperator.join(self._vTypes)
        addFilesString = "scenario/additionalsOutput_trafficMix_0_trafficDemand_1_driverBehaviour_OS_seed_0.xml, "+ vTypesToString + ", scenario/shapes.add.xml, scenario/view.add.xml"
        sumo_args = ["-c", self._cfg,
                    "-a", addFilesString]

        traci.start([sumolib.checkBinary('sumo')] + sumo_args)

        self.observation_space = spaces.Box(-100000, 100000,
                                            shape=(4, int(self.cellsPerAgent+2)), dtype=np.int)
        self.action_space = spaces.Discrete(self.cellsPerAgent+1)

        self.reward_range = (-float('inf'), float('inf'))
        self.run = 0
        self.myManager = TraciManager(self.cells_number,self.agents)
        self.plot = False
        self.seed = seed
        traci.close()


    def reset(self):
        """
        reset the enviromnent
        """
        self.start = time.time()
        if self.run != 0:
            traci.close()
        self.rsu_ids = list(range(self.agents))
        self.rsuActivatedCells = [0] * self.agents
        self.run += 1
        self.metrics = []
        self.x = []
        self.forcedT = 0
        self.tt = []
        self.ms = []
        self.total_reward = 0
        self.acted_times = 0

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
                        #  "--random",
                        "--seed", str(random.randint(1025, 2035)),
                        "-r", self._route,
                        "-a", addFilesString]

        traci.start([self._sumo_binary] + sumo_args)
        self.myManager = TraciManager(self.cells_number,self.agents)
        # self.myManager.do_steps(self.delay)

        return self._compute_observations()


    def _compute_observations(self):
        """
        Compute the observations arrays.
        It also store the TravelTime and MeanSpeed of Current step.
        """
        observations = {}
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

        # lv =  self.myManager.getLVperCells()
        # if not lv:
        #     lv = self.myManager.zerolistmaker(self.cells_number)

        speeds = self.myManager.getSpeedPerCells()
        if not speeds:
            speeds = self.myManager.zerolistmaker(self.cells_number)
        speeds = [round(x) for x in speeds]

        i = 0
        for rsu in self.rsu_ids:
            templ1 = []
            templ2 = []
            templ3 = []

            templ1 = av[i:int(self.cellsPerAgent+i+2)]
            templ2 = speeds[i:int(self.cellsPerAgent+i+2)]
            templ3 = pend[i:int(self.cellsPerAgent+i+2)]
            templ4 = self.myManager.ToC_Per_Cell[i:int(self.cellsPerAgent+i+2)]

            observations[rsu] = np.array(
                [templ1, templ2, templ3, templ4], dtype=np.int)
            i += self.cellsPerAgent
        
        return observations


    def step(self, action):
        """
        Execute environment step
        """
        # adapt action to cell number as the action 0 removed
        self._apply_actions(action)
        self._sumo_step()

        # observe new state and reward
        observation = self._compute_observations()
        reward = self._compute_rewards(action,observation)
        # self.total_reward += reward
        done =  self.myManager.sim_step() >= self.sim_max_time
        if(done):
            print()
            elapsed_time = time.time()- self.start
            print("Duration of the phase =", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
            if(self.sim_example==False):
                print("Train Episode " + str(self.run) + " of " + str(self.trains) +
                      " finished after " + str(self.myManager.sim_step()*10) + " timesteps")
            else:
                print("Simulation example finished after " +
                      str(self.myManager.sim_step()*10) + " timesteps")
            print("Number of ToC per cell")
            print(self.myManager.ToC_Per_Cell)

            print("Total number of ToC messages: " + str(sum(self.myManager.ToC_Per_Cell)))
            # print("CAVS at the last timestep: ")
            # print( str(observation[0]))
            print("Missed CAVS: ", str(int(self.forcedT)))
            # print("Number of forced ToC messages: " + str(self.myManager.forcedToCs))
            if(sum(self.myManager.ToC_Per_Cell) != 0):
                print("Average covered distance of CAV_CV vehs: " +
                      str(self.myManager.cav_dist/sum(self.myManager.ToC_Per_Cell)))
            print("Average Mean Speed in simulation for both of the lanes: " + str(sum(self.ms)/len(self.ms)))
            print("Average Travel Time in simulation for both of the lanes: " + str(sum(self.tt)/len(self.tt)))
            # print("Total Reward ", str(self.total_reward))
            # print("Number of Actions ", str(self.acted_times))
            # print("Number of Missed Actions ", str(self.sim_max_steps - self.delay - self.acted_times))
            print()
            if(self.sim_example):
                self.save_csv(self.data_path, self.run)



        done = {'__all__': self.myManager.sim_step() >= self.sim_max_time}

        info = self._compute_step_info(action, observation)
        self.metrics.append(info)
        self.last_reward = sum(reward.values())

        return observation, reward, done, {}


    def _apply_actions(self, actions):
        """
        Apply the actions in the environment
        More specifically store the activatedCell.
        """
        i = 0
        for rsu, action in actions.items():
            if(action == self.cellsPerAgent):
                self.rsuActivatedCells[rsu] = -1
                self.myManager.activatedCell[rsu] = -1
            else:
                self.rsuActivatedCells[rsu] = int(action+1+i)
                self.myManager.activatedCell[rsu] = int(action+1+i)
            i+=self.cellsPerAgent

        

    def _compute_rewards(self, action, observation):
        """
        We compute the rewards base on a specific function.
        2 alternatives
        """

        # reward = self.reward_based_on_DS_MA(action, observation)
        # reward = self.reward_based_cells_MA(action, observation)
        reward = self.reward_based_cells_MA_TEST(action, observation)


        return reward


    def reward_based_cells_MA_TEST(self, action, observation):
        """
        A reward focusing on distributin the ToRs based on cells influece.
        The last agent is always sending ToRs to CAVs that can see.
        Default mode for 14 cells.
        Related Punishment when the speed of activated cell in under limit.
        Punishment when there AV vehs in last 2 cells -> enables forced ToCs only for second Agent.
        Punishment when we choose cell but without sending ToCs.
        Appearence of WT -> huge negative rewards.
        """
        rewards = {}
        wt = 0
        lanes = self.myManager.getAreaLanes()
        for i in range(2):
            wt += self.myManager.getLaneWait(lanes[i])

        i = 0
        for rsu in self.rsu_ids:
            reward = 0
            pun = 0
            wtpun = 0
            action = self.rsuActivatedCells[rsu]

            if(action != -1):
                if(observation[rsu][1][action-1-i] < 15 and observation[rsu][1][action-1-i] > 0):
                    wtpun += 1
                if(sum(self.myManager.ToC_Per_Cell) == 0):
                    ratio = 1
                else:
                    ratio = self.myManager.ToC_Per_Cell[action-1] / sum(self.myManager.ToC_Per_Cell)
            
            if(rsu == self.agents-1):
                pun = observation[rsu][0][-1] +  observation[rsu][0][-2]

            if(wt == 0):
                if(pun == 0):
                    if(wtpun == 0):
                        if(action != -1):
                            if(rsu == self.agents-1):
                                Sum = np.sum(observation[rsu][0])
                                if( Sum == 0):
                                    reward = 1
                                else:
                                    reward = -int(Sum)
                            else:
                                # if(ratio <= self.myManager.getCellInfluence(action) and ratio > 0):
                                if(action == 2 or action == 1):
                                    # reward = self.myManager.getCellInfluence(action)
                                    reward = -1
                                else:
                                    if(self.myManager.ToC_Per_Cell[action-3] < self.myManager.ToC_Per_Cell[action-1]):
                                        # if(rsu == self.agents-1):
                                        #     print(self.myManager.ToC_Per_Cell[action-1])
                                        #     print(self.myManager.ToC_Per_Cell[action])
                                        reward = self.myManager.getCellInfluence(action)
                                    else:
                                        reward = -1
                        else:
                            if(rsu == self.agents-1):
                                reward = -10
                            else:
                                reward = self.myManager.getCellInfluence((rsu+2)*2)
                    else:
                        reward = -1
                else:
                    self.forcedT += len(self.myManager.missed)
                    self.myManager.sendForced()
                    reward = -100
            else:
                reward = -10000
            i += self.cellsPerAgent
            rewards[rsu] = float(reward)

        
        return rewards


    def reward_based_cells_MA(self, action, observation):
        """
        A reward focusing on distributin the ToRs based on cells influece.
        Default mode for 14 cells.
        Related Punishment when the speed of activated cell in under limit.
        Punishment when there AV vehs in last 2 cells -> enables forced ToCs only for second Agent.
        Punishment when we choose cell but without sending ToCs.
        Appearence of WT -> huge negative rewards.

        """
        rewards = {}
        wt = 0
        lanes = self.myManager.getAreaLanes()
        for i in range(2):
            wt += self.myManager.getLaneWait(lanes[i])

            i = 0
            for rsu in self.rsu_ids:
                reward = 0
                pun = 0
                wtpun = 0
                action = self.rsuActivatedCells[rsu]

                if(action != -1):
                    if(observation[rsu][1][action-1-i] < 15 and observation[rsu][1][action-1-i] > 0):
                        wtpun += 1
                    if(sum(self.myManager.ToC_Per_Cell) == 0):
                        ratio = 1
                    else:
                        ratio = self.myManager.ToC_Per_Cell[action -
                                                            1] / sum(self.myManager.ToC_Per_Cell)

                if(rsu == self.agents-1):
                    pun = observation[rsu][0][-1] + observation[rsu][0][-2]

                if(wt == 0):
                    if(pun == 0):
                        if(wtpun == 0):
                            if(action != -1):
                                # if(ratio <= self.myManager.getCellInfluence(action) and ratio > 0):
                                if(action == 12 or action == 11):
                                    reward = self.myManager.getCellInfluence(
                                        action)
                                else:
                                    if(self.myManager.ToC_Per_Cell[action-1] <= self.myManager.ToC_Per_Cell[action]):
                                        reward = self.myManager.getCellInfluence(
                                            action)
                                    else:
                                        reward = -1
                            else:
                                if(rsu == self.agents-1):
                                    reward = -1
                                else:
                                    reward = self.myManager.getCellInfluence(
                                        (rsu+2)*2)
                        else:
                            reward = -1
                    else:
                        self.forcedT += len(self.myManager.missed)
                        self.myManager.sendForced()
                        reward = -10
                else:
                    reward = -10000
                i += self.cellsPerAgent
                rewards[rsu] = float(reward)

        return rewards

    

    def reward_based_on_DS_MA(self, action, observation):
        """
        Multi Agent reward function based on the last Single Agent one.
        Default mode for 14 cells.
        WT related Punishment when the speed of activated cell in under limit.
        WT related Punishment when the speed sum of activated cell and next cell in under limit.
        Punishment when there AV vehs in last 2 cells -> enables forced ToCs only for second Agent.
        Punishment when we choose cell but without sending ToCs.
        Appearence of WT -> huge negative rewards.

        """
        rewards = {}
        wt = 0
        lanes = self.myManager.getAreaLanes()
        for i in range(2):
            wt += self.myManager.getLaneWait(lanes[i])

        i=0
        for rsu in self.rsu_ids:
            reward = 0
            pun = 0
            wtpun=0
            action = self.rsuActivatedCells[rsu]

            if(action !=-1):
                if(observation[rsu][1][action-1-i] < 15 and observation[rsu][1][action-1-i] > 0 and observation[rsu][1][action-1+2-i] > 0):
                    wtpun += 1
                if((observation[rsu][1][action-1-i]+observation[rsu][1][action-1+2-i]) < 40 and observation[rsu][1][action-1-i] > 0 and observation[rsu][1][action-1+2-i] > 0):
                    wtpun += 1
            if(rsu!=0):
                pun = observation[rsu][0][-1] + observation[rsu][0][-2]

            if(wt == 0):
                if(pun == 0):
                    if(wtpun == 0):
                        # self.acted_times += 1
                        if(action != -1):
                            if(observation[rsu][2][action-1-i] == 0 or self.myManager.getDecidedToCs() == 0):
                                reward = 0
                            else:
                                reward = self.myManager.getCellInfluence(action)
                        else:
                            if(rsu == self.agents-1):
                                reward = -1
                            else:
                                reward = self.myManager.getCellInfluence((rsu+2)*2)
                    else:
                        reward = -1
                else:
                    self.forcedT += len(self.myManager.missed)
                    self.myManager.sendForced()
                    reward = -10
            else:
                reward = -10000
            i += self.cellsPerAgent
            rewards[rsu] = float(reward)

        return rewards

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
            'avg_cav_dist' : cav_avg,
            'meanSpeed':  self.ms[-1],
            'TravelTime': self.tt[-1],
            'ForcedT': self.forcedT,
            'reward': float(self.last_reward),
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
