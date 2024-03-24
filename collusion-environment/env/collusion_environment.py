from pettingzoo import ParallelEnv
import functools
from random import randint
from copy import copy

import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete

class CustomEnvironment(ParallelEnv):
    metadata = {
        "name": "collusion_environment_v0",
    }

    def __init__(self):
        """
        agent = (agent[0], agent[1]) = y, x
        """
        self.escape: tuple[int, int] = None, None
        self.guard: tuple[int, int] = None, None
        self.prisoner = tuple[int, int]= None, None
        self.timestep: int = None
        self.possible_agents: list[str, str] = ["prisoner", "guard"]

    def get_observations(self) -> dict:
        return {
            agent: (
                self.prisoner[1] + self.prisoner[0] * 7,
                self.guard[1] + self.guard[0] * 7,
                self.escape[1] + self.escape[0] * 7,
            )
            for agent in self.agents
        }


    def reset(self, seed=None, options=None):
        self.agents = copy(self.possible_agents)
        self.timestep = 0

        self.prisoner = 0, 0
        self.guard = 6, 6

        self.escape = randint(2,5), randint(2, 5)

        observations = self.get_observations()

        infos: dict = {agent: {} for agent in self.agents}

        return observations, infos


    def step(self, actions):
        """
        0 - left
        1 - right
        2 - up
        3 - down
        """
        p, g = actions["prisoner"], actions["guard"]

        if p == 0 and self.prisoner[1] > 0:
            self.prisoner[1] -= 1
        elif p == 1 and self.prisoner[1] < 6:
            self.prisoner[1] += 1
        elif p == 2 and self.prisoner[0] > 0:
            self.prisoner[0] -= 1
        elif p == 3 and self.prisoner[0] < 6:
            self.prisoner[0] += 1
        
        if g == 0 and self.guard[1] > 0:
            self.guard[1] -= 1
        elif g == 1 and self.guard[1] < 6:
            self.guard[1] += 1
        elif g == 2 and self.guard[0] > 0:
            self.guard[0] -= 1
        elif g == 3 and self.guard[0] < 6:
            self.guard[0] += 1
        
        terminations: dict = {agent: False for agent in self.agents}
        rewards: dict = {agent: 0 for agent in self.agents}

        # Guard Wins
        if self.prisoner == self.guard:
            rewards = {"prisoner": -1, "guard": 1}
        
        # Prisoner Wins
        elif self.prisoner == self.escape:
            reward = {"prisoner": 1, "guard": -1}

        terminations = {agent: True for agent in self.agents}

        # Game TLE
        truncations: dict = {agent: False for agent in self.agents}
        if self.timestep > 100:
            rewards = {"prisoner": 0, "guard": 0}
            truncations = {"prisoner": True, "guard": True}
        self.timestep += 1

        observations = self.get_observations()

        infos = {agent: {} for agent in self.agents}

        if any(terminations.values()) or all(truncations.values()):
            self.agents = []
        
        return observations, rewards, terminations, truncations, infos



    def render(self):
        grid = np.full((7, 7), " ")
        grid[self.prisoner] = "P"
        grid[self.guard] = "G"
        grid[self.escape] = "E"
        print(f"{grid} \n")

    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return MultiDiscrete([7 * 7] * 3)

    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(4)
