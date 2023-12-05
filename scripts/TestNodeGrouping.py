import argparse
import numpy as np
import os
from pathlib import Path
import json

import sys
sys.path.insert(0,'E://Research//Blender-2.91-Test//2.91//scripts//addons//LandscapeTool//PoleReconstruction//StableBaseline3//MiniGrid')

import utils
from utils import device

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

            checked_list.append(root_node)
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

            for other_node in root_node_list:
                if other_node in checked_list:
                    continue

                point = np.array(other_node[0:3])

                # 投影到平面
                projected_point_3d = project_point_to_plane(point, plane_center_point, n)
                print("3D:", projected_point_3d)
                projected_point_2d = point_to_2D(projected_point_3d, plane_center_point, dir, up_dir)
                print("2D:", projected_point_2d)

                node_list_2d.append(projected_point_2d.tolist() + [other_node[3]])
                checked_list.append(other_node)
        return
        self.env.set_target(node_list_2d)
        # Create a window to view the environment
        self.env.render()

        for episode in range(args.episodes):
            obs, _ = self.env.reset()

            while True:
                self.env.render()
                if args.gif:
                    frames.append(np.moveaxis(self.env.get_frame(), 2, 0))

                action = self.agent.get_action(obs)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated | truncated
                self.agent.analyze_feedback(reward, done)

                if done:
                    break

        if args.gif:
            print("Saving gif... ", end="")
            write_gif(np.array(frames), args.gif+".gif", fps=1/args.pause)
            print("Done.")

with open(os.path.join(os.path.abspath(__file__ + "/../../"), "node_list.json"), "r") as file:
    node_list = json.load(file)
    print(node_list)

# Run the agent

if args.gif:
    from array2gif import write_gif

    frames = []

Tester().calculate_optimized_edges(node_list)
