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
        size=40,
        agent_start_pos=(5, 5),
        agent_start_dir=0,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 300

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            agent_view_size=17,
            **kwargs,
        )
        self.action_space = spaces.Discrete(3)
        self.node_list_2d = None
        self.offset_x = 0
        self.offset_y = 0
        self.root_pos = None
        self.test_idx = 0
        self.visited_array = None

        self.sample_parameters()

    @staticmethod
    def _gen_mission():
        return "grand mission"

    def set_target(self, node_list_2d, min_x, min_y,):
        self.node_list_2d = node_list_2d
        self.offset_x = abs(min_x) + 1
        self.offset_y = abs(min_y) + 1
        self.root_pos = [self.get_shift_x(self.offset_x), self.get_shift_y(self.offset_y)]

    def get_shift_x(self, old_x, interval=0.5):
        return round(old_x / interval) + 1

    def get_shift_y(self, old_y, interval=0.5):
        return self.height - 1 - (round(old_y / interval) + 1)

    def sample_parameters(self):
        import itertools
        parameter_list = [
            [0, 1],
            [-1, 0, 1],
            [-2, -1, 0],
            [-1, 0, 1],
            [-2, -1, 0]
        ]

        self.all_parameter_list = [c for c in itertools.product(*parameter_list)]

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        
        if self.node_list_2d == None:
            p = self.all_parameter_list[self.test_idx]
            if p[0] == 0:
                sample_points = [[0.5, 7.5], [6.5, 7.5], [3.5, 2]]
                idx = random.randint(0, 2)
                self.root_pos = sample_points[idx]
                self.root_pos[0] = self.get_shift_x(self.root_pos[0])
                self.root_pos[1] = self.get_shift_y(self.root_pos[1])
                self.node_list_2d = []
                self.node_list_2d.append([1, 7.5, 9])
                self.node_list_2d.append([2 + p[1] * 0.5, 7.5, 0])
                self.node_list_2d.append([6, 7.5, 9])
                self.node_list_2d.append([5 + p[2] * 0.5, 7.5, 0])
                self.node_list_2d.append([3.5, 5 + p[3] * 0.5, 1])
                self.node_list_2d.append([3.5, 4 + p[4] * 0.5, 10])
            elif p[0] == 1:
                sample_points = [[0.5, 7.5], [3.5, 2]]
                idx = random.randint(0, 1)
                self.root_pos = sample_points[idx]
                self.root_pos[0] = self.get_shift_x(self.root_pos[0])
                self.root_pos[1] = self.get_shift_y(self.root_pos[1])
                self.node_list_2d = []
                self.node_list_2d.append([1, 7.5, 9])
                self.node_list_2d.append([2 + p[1] * 0.5, 7.5, 0])
                self.node_list_2d.append([3.5, 5 + p[3] * 0.5, 1])
                self.node_list_2d.append([3.5, 4 + p[4] * 0.5, 10])
            self.test_idx += 1
            if self.test_idx >= len(self.all_parameter_list):
                self.test_idx = 0

        for i in range(1, height - 1):
            for j in range(1, width - 1):
                self.grid.set(j, i, Wall())

        for node in self.node_list_2d:
            # Place the goal
            goal_x = self.get_shift_x(node[0] + self.offset_x)
            goal_y = self.get_shift_y(node[1] + self.offset_y)
            category = node[2]
            self.goal = Goal()
            self.grid.set(goal_x, goal_y, self.goal)
         
            # Generate wall
            for i in range(1, height - 1):
                if i == goal_y or (category == 9) or self.grid.get(goal_x, i).type == 'goal':
                    continue
                self.grid.set(goal_x, i, Floor(color="grey"))
            for j in range(1, width - 1):
                if j == goal_x or (category == 10) or self.grid.get(j, goal_y).type == 'goal':
                    continue
                self.grid.set(j, goal_y, Floor(color="grey"))

        self.goal_count = len(self.node_list_2d)

        self.agent_dir = 0
        self.agent_pos = (self.root_pos[0], self.root_pos[1])
        self.node_list_2d = None

        # 紀錄Agent走的路徑
        self.visited_array = np.zeros(shape=(self.height, self.width))
        self.visited_array[self.root_pos[0]][self.root_pos[1]] = 1

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
                self.grid.set(fwd_pos[0], fwd_pos[1], Floor(color="grey"))
                self.visited_array[fwd_pos[0]][fwd_pos[1]] = 1
            if fwd_cell is not None and fwd_cell.type == "goal":
                reward = self._reward()
                self.goal_count -= 1
                
        else:
            raise ValueError(f"Unknown action: {action}")

        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        if self.goal_count <= 0:
            terminated = True
            reward = 10

        obs = self.gen_obs()

        return obs, reward, terminated, truncated, {}