from __future__ import annotations

import minigrid
from minigrid.wrappers import ImgObsWrapper
import gymnasium as gym
import torch
import torch.nn as nn

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall, Ball, Floor
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv

import random
import numpy as np

from gymnasium import spaces
import gymnasium as gym
gym.register(id='MyEnv-v0', entry_point='utils:EdgeEnv')

class EdgeEnv(MiniGridEnv):
    def __init__(
        self,
        size=10,
        agent_start_pos=(5, 5),
        agent_start_dir=0,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            agent_view_size=7,
            **kwargs,
        )
        self.action_space = spaces.Discrete(3)

    @staticmethod
    def _gen_mission():
        return "grand mission"

    def set_target(self, node_list_2d):
        print(node_list_2d)
        pass

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        
        # Place the goal
        goal_i1 = random.randint(1, 4)
        goal_j1 = random.randint(2, 4)
        self.goal1 = Goal()
        self.grid.set(goal_i1, goal_j1, self.goal1)

        goal_i2 = random.randint(6, 8)
        goal_j2 = random.randint(6, 8)
        self.goal2 = Goal()
        self.grid.set(goal_i2, goal_j2, self.goal2)

        self.goal_count = 2

        goal_i_list = [goal_i1, goal_i2]
        goal_j_list = [goal_j1, goal_j2]

        # Generate wall
        for i in range(1, width - 1):
            if i in goal_i_list:
                continue
            for j in range(1, height - 1):
                if j in goal_j_list:
                    continue
                self.grid.set(i, j, Wall())

        self.agent_dir = 0
        self.agent_pos = (goal_i1, goal_j2)

    def step(self, action):
        self.step_count += 1

        reward = 0
        terminated = False
        truncated = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = tuple(fwd_pos)
                self.grid.set(fwd_pos[0], fwd_pos[1], Floor(color="yellow"))
            if fwd_cell is not None and fwd_cell.type == "goal":
                self.grid.set(fwd_pos[0], fwd_pos[1], Floor(color="yellow"))
                reward = 0.1
                self.goal_count -= 1
                
        else:
            raise ValueError(f"Unknown action: {action}")

        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        if self.goal_count <= 0:
            terminated = True
            reward = self._reward()

        obs = self.gen_obs()

        return obs, reward, terminated, truncated, {}