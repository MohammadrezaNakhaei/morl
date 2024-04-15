import gym
from gym.spaces import box 
from environments.example_env import ExampleEnv
import numpy as np

import gym
from gym.spaces import box 
from environments.example_env import ExampleEnv


class CustomPendulum(ExampleEnv):
    def __init__(self, task=[0.5], n_tasks=2, **kwargs):
        self.tasks = np.array(task)
        self._goal = task[0] if task else 1 #
        self._goal = np.asarray(self._goal)
        self.n_tasks = n_tasks
        self._env = gym.make('Pendulum-v1', g=self.goal*10)
        self._max_episode_steps = self._env._max_episode_steps
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def get_task(self):
        return self._goal 
    
    def set_goal(self, goal):
        self._goal = np.asarray(goal)
        self._env.env.g = goal*10 # change the g in the wrapped env
        

    def set_all_goals(self, goals):
        assert self.n_tasks == len(goals)
        self.tasks = np.array(goals) 

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_task(self, idx=None):
        if idx is not None:
            self._goal = self.tasks[idx]
        self.reset()

    def reset(self,):
        self._env.reset()
        self._env.state[0] = np.pi
        self._env.state[1] = 0
        state = self._env.state
        th, thdot = state
        return np.array([np.cos(th), np.sin(th),thdot], dtype=np.float32)
    
    def step(self, action):
        return self._env.step(action)
    
    @property
    def goal(self,):
        return self._goal
    
