import argparse
import numpy as np
import os
from pathlib import Path
import json

import sys
sys.path.insert(0,'E://Research//Blender-2.91-Test//2.91//scripts//addons//LandscapeTool//PoleReconstruction//StableBaseline3//MiniGrid')

import utils
from utils import device
from PIL import Image

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env",
                    help="name of the environment to be run (REQUIRED)", default="MyEnv-v0")
parser.add_argument("--model",
                    help="name of the trained model (REQUIRED)", default="EdgeModel")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--shift", type=int, default=0,
                    help="number of times the environment is reset at the beginning (default: 0)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="select the action with highest probability (default: False)")
parser.add_argument("--pause", type=float, default=0.1,
                    help="pause duration between two consequent actions of the agent (default: 0.1)")
parser.add_argument("--gif", type=str, default=None,
                    help="store output as gif with the given filename")
parser.add_argument("--episodes", type=int, default=1,
                    help="number of episodes to visualize")
parser.add_argument("--memory", action="store_true", default=False,
                    help="add a LSTM to the model")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model")

args = parser.parse_args()

class Tester:
    def __init__(self):
        # Set seed for all randomness sources
        utils.seed(args.seed)

        # Set device

        print(f"Device: {device}\n")

        # Load environment

        self.env = utils.make_env(args.env, args.seed, render_mode="human")
        for _ in range(args.shift):
            self.env.reset()
        print("Environment loaded\n")

        # Load agent
        os.environ["RL_STORAGE"] = "E://Research//Blender-2.91-Test//2.91//scripts//addons//LandscapeTool//PoleReconstruction//StableBaseline3//MiniGrid//storage"
        model_dir = utils.get_model_dir(args.model)
        self.agent = utils.Agent(self.env.observation_space, self.env.action_space, model_dir,
                            argmax=args.argmax, use_memory=args.memory, use_text=args.text)
        print("Agent loaded\n")

    def dfs(self, agent_visited_array, x, y, visited, direction, edge, all_comps, result_edges):
        rows, cols = agent_visited_array.shape
        if (x < 0 or x >= rows) or (y < 0 or y >= cols) or agent_visited_array[x][y] == 0 or visited[x][y]:
            return

        # 標記當前點為visited
        visited[x][y] = True
        edge['Pts'].append((x, y))

        for comp in all_comps:
            if comp[0] == x and comp[1] == y:
                edge['Comps'].append(comp)
                all_comps.remove(comp)
                break

        # 檢查並繼續沿著目前方向探索
        if direction == 'right':
            self.dfs(agent_visited_array, x + 1, y, visited, direction, edge, all_comps, result_edges)
        elif direction == 'left':
            self.dfs(agent_visited_array, x - 1, y, visited, direction, edge, all_comps, result_edges)
        elif direction == 'up':
            self.dfs(agent_visited_array, x, y - 1, visited, direction, edge, all_comps, result_edges)
        elif direction == 'down':
            self.dfs(agent_visited_array, x, y + 1, visited, direction, edge, all_comps, result_edges)

        # 檢查是否有新分支，如果有，從這個新分支開始新的探索
        for dx, dy, new_direction in [(0, -1 ,'up'), (0, 1, 'down'), (-1, 0, 'left'), (1, 0, 'right')]:
            new_x, new_y = x + dx, y + dy
            if (0 <= new_x < rows) and (0 <= new_y < cols) and agent_visited_array[new_x][new_y] == 1 and not visited[new_x][new_y]:
                new_edge = {
                    'Pts': [(x, y)],
                    'Comps': []
                }
                self.dfs(agent_visited_array, new_x, new_y, visited, new_direction, new_edge, all_comps, result_edges)
                if len(new_edge['Pts']) > 1:
                    result_edges.append(new_edge)

    def explore_ones(self, agent_visited_array, start_x, start_y, offset_x, offset_y, all_comps):
        rows, cols = agent_visited_array.shape
        visited = [[False for _ in range(cols)] for _ in range(rows)]

        root_edges = [{'Pts': [], 'Comps': []} for i in range(4)]

        result_edges = []
        for comp in all_comps:
            print(comp)

        root_comp = all_comps[-1]
        visited[start_x][start_y] = True

        print("-----------------------------------")
        self.dfs(agent_visited_array, start_x - 1, start_y, visited, 'left', root_edges[0], all_comps, result_edges)
        self.dfs(agent_visited_array, start_x + 1, start_y, visited, 'right', root_edges[1], all_comps, result_edges)
        self.dfs(agent_visited_array, start_x, start_y - 1, visited, 'up', root_edges[2], all_comps, result_edges)
        self.dfs(agent_visited_array, start_x, start_y + 1, visited, 'down', root_edges[3], all_comps, result_edges)

        for root_edge in root_edges:
            if len(root_edge['Pts']) > 1:
                root_edge['Comps'].append(root_comp)
                result_edges.append(root_edge)

        result = {
            'Root Pos': [start_x, start_y],
            'Offset': [offset_x, offset_y],
            'Result Edges': result_edges
        }

        with open(os.path.join(os.path.abspath(__file__ + "/../../"), "result.json"), "w") as file:
            json.dump(result, file)
        #edge['Pts'].clear()

    def calculate_optimized_edges(self, node_list):
        def point_to_2D(point, plane_point, v1, v2):
            translated_point = point - plane_point

            x = np.dot(translated_point, v1)
            y = np.dot(translated_point, v2)
            return np.array([x, y])

        def project_point_to_plane(point, plane_point, plane_normal):
            # 計算點和平面上一點的向量
            v = point - plane_point
            
            # 計算該向量在法向量上的投影長度
            dist = np.dot(v, plane_normal)
            
            # 計算投影點
            projection = point - dist * plane_normal
            return projection

        # 尋找Root Node
        root_priority_list = [9, 0, 1, 10, 11]  # 按照指定類型來排序 node list
        root_node_list = sorted(node_list, key=lambda x: root_priority_list.index(x[3]) if x[3] in root_priority_list else float('inf'))
        checked_list = []

        for root_node in root_node_list:
            if root_node in checked_list:
                continue

            plane_center_point = np.array(root_node[0:3])
            node_list_2d = []
            print("Root:", plane_center_point)
            # 取得整個Grid要生長的方向
            edge_angle = root_node[5]

            theta = np.radians(edge_angle)

            # 創建旋轉矩陣
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])

            # 進行座標旋轉
            up_dir = np.array([0, 0, 1])
            dir = rotation_matrix.dot(np.array([1, 0, 0])).tolist()
            n = np.cross(up_dir, dir)
            min_x = 0
            min_y = 0

            for other_node in root_node_list:
                if other_node in checked_list:
                    continue

                point = np.array(other_node[0:3])

                # 投影到平面
                projected_point_3d = project_point_to_plane(point, plane_center_point, n)
                print("3D:", projected_point_3d)
                projected_point_2d = point_to_2D(projected_point_3d, plane_center_point, dir, up_dir)
                print("2D:", projected_point_2d)

                node_list_2d.append(projected_point_2d.tolist() + [other_node[3]] + [other_node[4]])
                checked_list.append(other_node)
                
                if projected_point_2d[0] < min_x:
                    min_x = projected_point_2d[0]

                if projected_point_2d[1] < min_y:
                    min_y = projected_point_2d[1] 
        
            self.env.set_target(node_list_2d, min_x, min_y)
            # Create a window to view the environment
            self.env.render()

            for episode in range(args.episodes):
                obs, _ = self.env.reset()
                frames = []
                while True:
                    self.env.render()
                    if args.gif:
                        frames.append(self.env.get_frame().astype(np.uint8))

                    action = self.agent.get_action(obs)
                    obs, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated | truncated
                    self.agent.analyze_feedback(reward, done)
                    
                    if done:
                        all_comps = self.env.target_obejcts.copy()
                        # 把Root元件加入List
                        all_comps.append((self.env.root_pos[0], self.env.root_pos[1], root_node[4]))
                        print("Visited Array")
                        np.set_printoptions(linewidth=200)
                        for e in self.env.visited_array:
                            print(e)
                        self.explore_ones(
                            self.env.visited_array, 
                            self.env.root_pos[0], 
                            self.env.root_pos[1], 
                            self.env.offset_x,
                            self.env.offset_y,
                            all_comps)
                        break

            if args.gif:
                print("Saving gif... ", end="")
                imgs = [Image.fromarray(img).quantize(method=Image.MEDIANCUT) for img in frames]
                # duration is the number of milliseconds between frames
                imgs[0].save(os.path.join(os.path.abspath(__file__ + "/../../gifs/"), args.gif + f"{episode}.gif"), save_all=True, append_images=imgs[1:], duration=200, loop=0)
                print("Done.")

with open(os.path.join(os.path.abspath(__file__ + "/../../"), "node_list.json"), "r") as file:
    node_list = json.load(file)
    print(node_list)

# Run the agent

if args.gif:
    from array2gif import write_gif

    frames = []

Tester().calculate_optimized_edges(node_list)
