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
import os

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
            max_steps = 400

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            agent_view_size=25,
            **kwargs,
        )
        self.action_space = spaces.Discrete(3)
        self.node_list_2d = None
        self.offset_x = 0
        self.offset_y = 0
        self.root_pos = None
        self.test_idx = 0
        self.visited_array = None
        self.target_obejcts = None  # 紀錄元件在Grid的對應座標
        self.visited_count = 0
        self.all_visited_count = 0

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
            [0, 1, 2], # 起點
            [-1, 0, 1],
            [-2, -1, 0],
            [-1, 0, 1],
            [-2, -1, 0],
            [2], # 類型
        ]

        self.all_parameter_list = [c for c in itertools.product(*parameter_list)]

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        
        if self.node_list_2d == None:
            p = self.all_parameter_list[self.test_idx]
            if p[5] == 0:   # 雙桿紅綠燈
                sample_points = [[0.5, 7.5], [6 - p[2] + 0.5, 7.5], [3.5 - p[2], 4 + p[4] * 0.5 - 0.5]]
                idx = p[0]
                self.root_pos = sample_points[idx]
                self.root_pos[0] = self.get_shift_x(self.root_pos[0])
                self.root_pos[1] = self.get_shift_y(self.root_pos[1])
                self.node_list_2d = []
                self.node_list_2d.append([1, 7.5, 9])
                self.node_list_2d.append([2 + p[1] * 0.5 - p[2], 7.5, 0])
                self.node_list_2d.append([6 - p[2], 7.5, 9])
                self.node_list_2d.append([5 + p[2] * 0.5 - p[2], 7.5, 0])
                self.node_list_2d.append([3.5 - p[2], 5 + p[3] * 0.5, 1])
                self.node_list_2d.append([3.5 - p[2], 4 + p[4] * 0.5, 10])
            elif p[5] == 1: # 單桿紅綠燈
                sample_points = [[0.5, 7.5], [3.5 - p[2], 4 + p[4] * 0.5 - 0.5], [3.5 - p[2], 7.5]]
                idx = p[0]
                self.root_pos = sample_points[idx]
                self.root_pos[0] = self.get_shift_x(self.root_pos[0])
                self.root_pos[1] = self.get_shift_y(self.root_pos[1])
                self.node_list_2d = []
                self.node_list_2d.append([1 - p[2], 7.5, 9])
                self.node_list_2d.append([2 + p[1] * 0.5 - p[2], 7.5, 0])
                self.node_list_2d.append([3.5 - p[2], 5 + p[3] * 0.5, 1])
                self.node_list_2d.append([3.5 - p[2], 4 + p[4] * 0.5, 10])
            elif p[5] == 2:
                sample_points = [[2 + p[1] * 0.5 - p[2] + 0.5, 7.5], [4 + p[3] * 0.5 - p[2] + 0.5, 7.5], [7 - p[2], 4 + p[4] * 0.5 - 0.5]]
                idx = p[0]
                self.root_pos = sample_points[idx]
                self.root_pos[0] = self.get_shift_x(self.root_pos[0])
                self.root_pos[1] = self.get_shift_y(self.root_pos[1])
                self.node_list_2d = []
                self.node_list_2d.append([1, 4 + p[4] * 0.5, 10])
                self.node_list_2d.append([7 - p[2], 4 + p[4] * 0.5, 10])
                self.node_list_2d.append([2 + p[1] * 0.5 - p[2], 7.5, 0])
                self.node_list_2d.append([4 + p[3] * 0.5 - p[2], 7.5, 1])
                self.node_list_2d.append([6 + p[1] * 0.5 - p[2], 7.5, 0])
            self.test_idx += 1
            if self.test_idx >= len(self.all_parameter_list):
                self.test_idx = 0

        for i in range(1, height - 1):
            for j in range(1, width - 1):
                self.grid.set(j, i, Wall())

        self.target_obejcts = []
        for node in self.node_list_2d:
            # Place the goal
            goal_x = self.get_shift_x(node[0] + self.offset_x)
            goal_y = self.get_shift_y(node[1] + self.offset_y)
            category = node[2]
            self.goal = Goal()
            self.grid.set(goal_x, goal_y, self.goal)
            if len(node) >= 4:
                self.target_obejcts.append((goal_x, goal_y, node[3]))
         
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

        self.visited_count = 0
        self.all_visited_count = 0

        for i in range(1, height - 1):    
            for j in range(1, width - 1):
                if self.grid.get(j, i).can_overlap():
                    self.all_visited_count += 1

    def _reward(self):
        visited_term =  1 - 0.9 * (self.visited_count * 1.0 / self.all_visited_count)
        step_term = 1 - 0.9 * (self.step_count / self.max_steps)
        return step_term

    def is_cross(self, current_pos):
        x, y = current_pos
        up_cell = self.grid.get(x, y - 1).can_overlap() if y - 1 >= 0 else False
        down_cell = self.grid.get(x, y + 1).can_overlap() if y + 1 < self.height else False
        left_cell = self.grid.get(x - 1, y).can_overlap() if x - 1 >= 0 else False
        right_cell = self.grid.get(x + 1, y).can_overlap() if x + 1 < self.width else False

        return (up_cell or down_cell) and (left_cell or right_cell)

    def get_best_forward_pos(self):
        '''
        計算Forward要走到哪個點
        如果有Goal就停在最遠的Goal
        如果沒有Goal就停在最近的交叉點
        都沒有就設定成最遠的可站點
        '''

        current_pos = np.array(self.agent_pos, dtype=np.int8)
        next_pos = np.array(self.front_pos, dtype=np.int8)
        offset = next_pos - current_pos
        check_pos = current_pos
        farest_goal = None
        nearest_cross = None
        farest_ovrelap_pos = None

        while True:
            check_cell = self.grid.get(*check_pos)
            if check_cell is not None:
                if check_cell.type == "goal":
                    farest_goal = np.array(check_pos)
                elif self.is_cross(check_pos):
                    nearest_cross = np.array(check_pos)
                elif check_cell.can_overlap():
                    farest_ovrelap_pos = np.array(check_pos)
                else:
                    break
            check_pos = check_pos + offset

        if farest_goal is not None:
            return farest_goal, offset
        elif nearest_cross is not None:
            return nearest_cross, offset
        else:
            return farest_ovrelap_pos, offset

    def step(self, action):
        self.step_count += 1

        reward = 0
        terminated = False
        truncated = False

        # 計算Forward要走到哪個點
        best_forward_pos, offset = self.get_best_forward_pos()

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
            if fwd_cell.can_overlap():
                check_pos = np.array(self.front_pos)
                while True:
                    print(check_pos)
                    check_cell = self.grid.get(*check_pos)
                    if check_cell.type == "goal":
                        self.grid.set(check_pos[0], check_pos[1], Floor(color="blue"))
                        reward = self._reward()
                        self.goal_count -= 1
                    if self.visited_array[check_pos[0]][check_pos[1]] != 1:
                        self.visited_count += 1
                        self.visited_array[check_pos[0]][check_pos[1]] = 1
                    if np.array_equal(check_pos, best_forward_pos):
                        break
                    
                    check_pos = check_pos + offset
                    if check_pos[0] < 0 or check_pos[0] >= self.width or check_pos[1] < 0 or check_pos[1] >= self.height:
                        break

                self.agent_pos = tuple(best_forward_pos)
            else:
                reward = -1
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