import gym
from gym.spaces import box 
from environments.example_env import ExampleEnv


import gym
from gym.spaces import box 
from environments.example_env import ExampleEnv


class CustomPendulum(ExampleEnv):
    def __init__(self, task=[], n_tasks=2):
        self.tasks = task
        self.goal = task[0] if task else 1.0 #
        self.n_tasks = n_tasks
        self._env = gym.make('Pendulum-v1', g=self.goal*10)
        self._max_episode_steps = self._env._max_episode_steps
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def get_task(self):
        return self.goal 
    
    def set_goal(self, goal):
        self.goal = goal
        self._env.env.g = goal*10 # change the g in the wrapped env

    def set_all_goals(self, goals):
        assert self.n_tasks == len(goals)
        self.tasks = goals 

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_task(self, idx=None):
        if idx is not None:
            self._task = self.tasks[idx]
            self._goal_vel = self._task['velocity']
            self._goal = self._goal_vel
        self.reset()

    def reset(self,):
        return self._env.reset()
    
    def step(self, action):
        return self._env.step(action)