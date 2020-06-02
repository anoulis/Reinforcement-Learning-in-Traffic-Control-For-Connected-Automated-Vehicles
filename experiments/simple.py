import os
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # temporaly disable gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # see gpu

import gym
import numpy as np

import tensorflow as tf
from datetime import datetime
import random
from collections import deque
from stable_baselines.deepq import DQN, MlpPolicy, LnCnnPolicy, LnMlpPolicy
from stable_baselines.common.env_checker import check_env
from stable_baselines.deepq.policies import DQNPolicy
from stable_baselines.common.policies import CnnLnLstmPolicy
from stable_baselines.common.buffers import ReplayBuffer, PrioritizedReplayBuffer
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import A2C

import argparse
import sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import pandas as pd
from gym import spaces
import numpy as np
import traci

def main():
    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")
    print('tensorflow version', tf.__version__)
    print(gym.__version__)
    parser = argparse.ArgumentParser(description='Process some entries.')

    parser.add_argument("-cfg", dest="cfg", type=str,
                        default='scenario/sumo.cfg',
                        help="Network definition xml file.\n")
    parser.add_argument("-net", dest="network", type=str,
                        default='scenario/UC5_1.net.xml',
                        help="Network definition xml file.\n")
    parser.add_argument("-route", dest="route", type=str,
                        default='scenario/routes_trafficMix_0_trafficDemand_1_driverBehaviour_OS_seed_0.xml',
                        help="Route definition xml file.\n")
    parser.add_argument("-vTypes", dest="vTypes", type=str, nargs='*',
                        default=['scenario/vTypesCAVToC_OS.add.xml','scenario/vTypesCVToC_OS.add.xml','scenario/vTypesLV_OS.add.xml'],
                        help="Route definition xml file.\n")
    parser.add_argument("-gui", action="store_true", default=True, help="Run with visualization on SUMO.\n"),
    parser.add_argument("-sim_steps", dest="sim_steps", type =int, default=4000, help="Max simulation steps.\n"),
    parser.add_argument("-trains", dest="trains", type =int, default=30, help="Max trainings.\n"),


    # parser.add_argument("-runs", dest="runs", type=int, default=1, help="Number of runs.\n")

    args = parser.parse_args()
    experiment_time = str(datetime.now()).split('.')[0]

    env = gym.make('tor_distribution:tor-v0',
                    cfg_file=args.cfg,
                    net_file=args.network,
                    route_file=args.route,
                    vTypes_files=args.vTypes,
                    use_gui=args.gui,
                    sim_steps = args.sim_steps,
                    num_seconds=10000,
                    max_depart_delay=0)

    # It will check your custom environment and output additional warnings if needed
    # check_env(env)

    # # initialization of the DQN training model
    # model = DQN(
    #     env=env,
    #     policy=LnMlpPolicy,
    #     gamma=0.99,
    #     prioritized_replay=True,
    #     learning_rate=1e-3,
    #     buffer_size=50000,
    #     exploration_fraction=0.1,
    #     exploration_final_eps=0.02,
    #     verbose = 1,
    #     tensorboard_log="./dqn_tensorboard/"
    # )
    #
    # # execute the training
    # model.learn(total_timesteps=(args.trains*args.sim_steps))
    #
    # # save, delete and restore model
    # model.save("dnq_sample")
    # del model
    model = DQN.load("dnq_sample")

    # # initialization of the A2C training model
    # model = A2C(MlpPolicy, env, verbose=0, learning_rate=0.0001, lr_schedule='constant')
    #
    # # execute the training
    # model.learn(total_timesteps=(args.trains*args.sim_steps))
    #
    # # save, delete and restore model
    # model.save("a2c_sample")
    # del model
    # model = A2C.load("a2c_sample")

    if(args.gui):
        env.render()
    observation = env.reset()
    for t in range(100000):
        # action = env.action_space.sample()
        action, _states = model.predict(observation)
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            print("reward " + str(reward))
            print(info.get("avg_cav_dist"))
            break
    env.close()

if __name__ == '__main__':
    main()
