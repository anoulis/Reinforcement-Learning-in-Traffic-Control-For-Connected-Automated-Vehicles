from gym.envs.registration import register

register(
    id='tor-v0',
    entry_point='tor_distribution.envs:TorEnv',
)
