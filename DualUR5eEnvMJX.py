import jax
import jax.numpy as jnp
from gym import spaces
from dm_control.mujoco import wrapper
import numpy as np


class DualUR5eEnvMJX:
    def __init__(self, model_path, frame_skip=1):
        # Load the MuJoCo model
        self.model = wrapper.MjModel.from_xml_path(model_path)
        self.data = wrapper.MjData(self.model)

        self.frame_skip = frame_skip

        # Define observation and action spaces
        self.obs_size = self.model.nq + self.model.nv
        self.observation_space = spaces.Box(
            low=-jnp.inf, high=jnp.inf, shape=(self.obs_size,), dtype=jnp.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.model.nu,), dtype=jnp.float32
        )

    def reset(self, key):
        """
        Reset the environment state.
        """
        random_state = jax.random.uniform(key, shape=(self.obs_size,), minval=-1.0, maxval=1.0)

        # Convert JAX array to NumPy before assignment
        # self.data.qpos[:] = np.array(random_state[: self.model.nq])
        # self.data.qvel[:] = np.array(random_state[self.model.nq:])
        obs = self._get_obs()
        return obs

    def step(self, state, action):
        """
        Perform a step in the environment.
        """
        for _ in range(self.frame_skip):
            self.data.ctrl = action
            self.model.step(self.data)

        obs = self._get_obs()
        reward = self._compute_reward()
        done = False
        return obs, reward, done

    def _get_obs(self):
        """
        Get the current observation.
        """
        return jnp.concatenate([self.data.qpos.ravel(), self.data.qvel.ravel()])

    def _compute_reward(self):
        """
        Compute the reward for the current step.
        """
        return 1.0  # Dummy reward, replace with your task-specific logic
