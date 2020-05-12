import os
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import gym
import numpy as np
import tensorflow as tf
from datetime import datetime

from stable_baselines.deepq import DQN, MlpPolicy
from stable_baselines.common.env_checker import check_env

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
# from sumo_rl.environment.env import SumoEnvironment
import traci

def main():
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
    parser.add_argument("-runs", dest="runs", type=int, default=1, help="Number of runs.\n")



    args = parser.parse_args()
    experiment_time = str(datetime.now()).split('.')[0]

    env = gym.make('tor_distribution:tor-v0',
                    cfg_file=args.cfg,
                    net_file=args.network,
                    route_file=args.route,
                    vTypes_files=args.vTypes,
                    use_gui=args.gui,
                    num_seconds=10000,
                    max_depart_delay=0)
    print("Observation space:", env.observation_space)


    # It will check your custom environment and output additional warnings if needed
    check_env(env)

    model = DQN(
        env=env,
        policy=MlpPolicy,
        learning_rate=1e-3,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02
    )

    model.learn(total_timesteps=100000)

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        # env.render()
        if done:
            model.save("out.csv")
            break

    # for episode in range(self.num_of_episodes+1):
    #     state = env.reset()
    #     state = self.state_reshape(state)
    #     r = []
    #     t = 0
    #     while True:
    #         action = model.act(state)
    #         next_state, reward, done, _ = env.step(action)
    #         next_state = self.state_reshape(next_state)
    #         self.remember(state, next_state, action, reward, done)
    #         state = next_state
    #         r.append(reward)
    #         t += 1
    #         if done:
    #             r = np.mean(r)
    #             print("episode number: ", episode,", reward: ",r , "time score: ", t)
    #             self.save_info(episode, r, t)
    #             break
    #     self.replay()



    # for run in range(1, args.runs+1):
    #     initial_states = env.reset()
    #
    #
    # model.learn(total_timesteps=100000)
    #
    # if render: env.render()
    #
    # state, rew, reset = env.reset(), 0, False
    # while not reset:
    #     state, rew, reset, _ = env.step(action=agent.act(state, rew, reset))
    # agent.episode_end(env.env_id)
    # env.close()
    #
    # obs = env.reset()
    # for i in range(1000):
    #     action, _states = model.predict(obs)
    #     obs, rewards, dones, info = env.step(action)
    #     env.render()


if __name__ == '__main__':
    main()
