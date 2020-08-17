from ray.rllib.utils.annotations import override
import numpy as np
import random
from ray.rllib.policy.policy import Policy
from ray.rllib.agents.dqn.dqn_tf_policy import DQNTFPolicy
import tensorflow as tf
from random import randrange
from gym.spaces import Discrete
import numpy as np
import ray
from ray.rllib.agents.dqn.distributional_q_tf_model import DistributionalQTFModel
from ray.rllib.agents.dqn.simple_q_tf_policy import TargetNetworkMixin
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_action_dist import Categorical
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import LearningRateSchedule
from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils.exploration import ParameterNoise
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.tf_ops import huber_loss, reduce_mean_ignore_inf, minimize_and_clip
from ray.rllib.utils.tf_ops import make_tf_callable
from ray.rllib.agents.dqn import dqn_tf_policy
from ray.rllib.examples.models.custom_loss_model import CustomLossModel

tf = try_import_tf()

Q_SCOPE = "q_func"
Q_TARGET_SCOPE = "target_q_func"

# Importance sampling weights for prioritized replay
PRIO_WEIGHTS = "weights"

class RandomPolicy(Policy):
    """Hand-coded policy that returns random actions."""

    def __init__(self, *args, **kwargs):
        print(**kwargs)
        super().__init__(*args, **kwargs)

    @override(DQNTFPolicy)
    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        **kwargs):
        # Alternatively, a numpy array would work here as well.
        # e.g.: np.array([random.choice([0, 1])] * len(obs_batch))
        # print(self.action_space.sample())
        return [self.action_space.sample() for _ in obs_batch], [], {}

    @override(DQNTFPolicy)
    def learn_on_batch(self, samples):
        """No learning."""
        return {}
    
    def rmsprop(self):
        tf.keras.optimizers.RMSprop(
            learning_rate=0.001, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False,
            name='RMSprop')

    @override(DQNTFPolicy)
    def compute_log_likelihoods(self,
                                actions,
                                obs_batch,
                                state_batches=None,
                                prev_action_batch=None,
                                prev_reward_batch=None):
        return np.array([random.random()] * len(obs_batch))
    
ModelCatalog.register_custom_model("custom_loss", CustomLossModel)

def RMSProp_optimizer(policy, config):
    return tf.train.RMSPropOptimizer(policy.cur_lr, 0.99,0.0, 0.1)



myDQNTFPolicy = build_tf_policy(
    name="myDQNTFPolicy",
    get_default_config=lambda: ray.rllib.agents.dqn.dqn.DEFAULT_CONFIG,
    make_model=dqn_tf_policy.build_q_model,
    action_distribution_fn=dqn_tf_policy.get_distribution_inputs_and_class,
    loss_fn=dqn_tf_policy.build_q_losses,
    stats_fn=dqn_tf_policy.build_q_stats,
    postprocess_fn=dqn_tf_policy.postprocess_nstep_and_prio,
    optimizer_fn=RMSProp_optimizer,
    gradients_fn=dqn_tf_policy.clip_gradients,
    extra_action_fetches_fn=lambda policy: {"q_values": policy.q_values},
    extra_learn_fetches_fn=lambda policy: {"td_error": policy.q_loss.td_error},
    before_init=dqn_tf_policy.setup_early_mixins,
    before_loss_init=dqn_tf_policy.setup_mid_mixins,
    after_init=dqn_tf_policy.setup_late_mixins,
    obs_include_prev_action_reward=False,
    mixins=[
        TargetNetworkMixin,
        dqn_tf_policy.ComputeTDErrorMixin,
        LearningRateSchedule,
    ])
