import ray
from ray.rllib.agents.a3c.a3c import A3CTrainer
from ray.rllib.agents.a3c.a3c_tf_policy import A3CTFPolicy
from ray.tune.registry import register_env
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.agents.dqn.dqn_tf_policy import DQNTFPolicy
from ray.rllib.examples.policy.random_policy import RandomPolicy
from ray.tune.logger import pretty_print
from ray import tune
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.evaluation.postprocessing import discount
from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.utils.framework import try_import_tf
from myDQNTFPolicy import RandomPolicy, myDQNTFPolicy
import subprocess
from ray.rllib.examples.models.custom_loss_model import CustomLossModel
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.models import ModelCatalog
from time import strftime
from copy import deepcopy
import os
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # temporaly disable gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # see gpu

import gym
gym.logger.set_level(40)
import numpy as np
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import tensorflow as tf
from datetime import datetime
import random
from collections import deque
from tor_distribution.envs.tor_env import TorEnv
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
import time
import os
from datetime import datetime
from callbacks import MyCallbacks
import tempfile
from  ray.tune.logger import UnifiedLogger

DEFAULT_RESULTS_DIR = '/home/anoulis/ray_results'


def main():
    start = time.time()
    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF or use the GPU")

    tf.compat.v1.logging.set_verbosity( tf.compat.v1.logging.ERROR)

    print('tensorflow version', tf.__version__)
    print('gym version', gym.__version__)
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
    parser.add_argument("-plot", action="store_true", default=True, help="Plot graphs.\n"),
    parser.add_argument("-sim_steps", dest="sim_steps", type =int, default=48335 , help="Max simulation steps.\n"),
    parser.add_argument("-trains", dest="trains", type =int, default=30, help="Max trainings.\n"),
    parser.add_argument("-pun", dest="pun", type =float, default=1.0, help="Forced ToC messages punishment factor.\n"),
    parser.add_argument("-zip", dest="zip", type=str,
                        default='dqn_sample.zip',
                        help="Load the dqn model zip file.\n")
    parser.add_argument("-mode", dest="mode", type=str,
                        default='train',
                        help="Train or Eval\n")
    parser.add_argument("-simulations", dest="simulations", type=int,
                        default=10, help="Number of simulation examples.\n"),


    # parser.add_argument("-runs", dest="runs", type=int, default=1, help="Number of runs.\n")

    args = parser.parse_args()
    experiment_time = str(datetime.now()).split('.')[0]

    # data folder for every experiment
    path = os.getcwd() + "/outputs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    # path = os.getcwd() + "/outputs/trainings/" + args.zip+"_"+datetime.now().strftime("%Y%m%d-%H%M%S")

    delay = 0
    envName = "tor-v0"
    eval_path = ""

    
    if args.mode == 'train':
            # Register the model and environment
            register_env(envName, lambda _: TorEnv(
                            cfg_file=args.cfg,
                            net_file=args.network,
                            route_file=args.route,
                            vTypes_files=args.vTypes,
                            use_gui=args.gui,
                            sim_steps=args.sim_steps,
                            trains = args.trains,
                            plot = args.plot,
                            delay=delay,
                            forced_toc_pun=args.pun,
                            data_path = path))
            
            try:
                do_training(envName, args.sim_steps, args.trains)
            except AssertionError as error:
                print(error)
                print("Problem on training")

    elif args.mode == 'eval':
        print("Let's test")
        rollout('/home/anoulis/ray_results/'+eval_path,
                envName, args.sim_steps, args.simulations)

def policy_mapping_fn(agent_id):
    # if agent_id == 0:
    #     return "dqn_policy"
    # else:
    #     return "mydqn_policy"
    return "dqn_policy"


def do_training(envName, sim_steps, trains):
    """Train policies using the DQN algorithm in RLlib."""

    obs_space = spaces.Box(low=-1000, high=1000,shape=(4, 8), dtype=np.int)
    act_space = spaces.Discrete(6)

    policies = {
        'dqn_policy': (DQNTFPolicy, obs_space, act_space, {}),
        "mydqn_policy": (DQNTFPolicy, obs_space, act_space, {}),
    }

    ModelCatalog.register_custom_model("custom_loss", CustomLossModel)

    n_cpus = 0
    labels = n_cpus
    # policies_to_train = ['dqn_policy']

    ray.init(num_cpus=n_cpus + 1, memory=6000 * 1024 * 1024,
             object_store_memory=3000 * 1024 * 1024)

    myConfig = {

        # Mutli Agent Configs
        "multiagent": {
            # "model": {
            #     "custom_model": "custom_loss",
            #     "use_lstm": True,
            #     "lstm_use_prev_action_reward": True,
            # },
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn,
            # "policies_to_train": policies_to_train
        },

        # DQN params 
        "lr": 0.0005,
        "target_network_update_freq": 100,
        "buffer_size": 300000,

        # Train-Sim Params
        'timesteps_per_iteration':sim_steps,
        "framework": "tfe",
        'eager_tracing': True,
        "num_workers": n_cpus,
        "callbacks": MyCallbacks,
    }
    timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    logdir_prefix = "{}__{}".format('DQN',timestr)

    def default_logger_creator(config):
        """Creates a Unified logger with a default logdir prefix
        containing the agent name and the env id
        """
        if not os.path.exists(DEFAULT_RESULTS_DIR):
            os.makedirs(DEFAULT_RESULTS_DIR)
        logdir = tempfile.mkdtemp(
            prefix=logdir_prefix, dir=DEFAULT_RESULTS_DIR)
        return UnifiedLogger(config, logdir, loggers=None)


    trainer = DQNTrainer(env=envName, config=myConfig,
                         logger_creator=default_logger_creator)
    # if logger_creator is None:
    #     timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    #     logdir_prefix = "{}_{}_{}".format(self._name, self._env_id,
    #                                       timestr)
    #     logger_creator = default_logger_creator()

    for i in range(trains):
        print(f'== Training Iteration {i+1}==')
        print(pretty_print(trainer.train()))
        checkpoint = trainer.save()
        print(f'\nCheckpoint saved at {checkpoint}\n')

    try:
        TorEnv.close(TorEnv)
    except:
        pass
    ray.shutdown()


def rollout(checkpoint_path, envName, sim_steps, simulations):
    subprocess.call([
        sys.executable,
        './rollout.py', checkpoint_path,
        '--env', envName,
        '--steps', str(sim_steps),
        '--run', 'DQN',
        '--no-render',
        '-sim_steps', str(sim_steps),
        '-simulations', str(simulations)
    ])

if __name__ == '__main__':
    main()
