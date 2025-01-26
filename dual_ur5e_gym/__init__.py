from gym.envs.registration import register
from dual_ur5e_gym.version import VERSION as __version__

register(
    id="DualUR5eEnv",
    entry_point="dual_ur5e_gym.envs:DualUR5eEnv",
)
register(
    id="DualUR5eEnvMJX",
    entry_point="dual_ur5e_gym.envs:DualUR5eEnvMJX",
)
