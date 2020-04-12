# import deep_learning_rl as Simulator  
from SimulationEnvironment.PutGroceriesEnv import PutGrocceriesRLGraspingEnvironment
from models.CupboardAgents.GripperPoseAgent import GripperPoseRLAgent    
import time
# Set dataset_root to load from a folder and datasst will load from there. 
curr_env = PutGrocceriesRLGraspingEnvironment(dataset_root='/home/valay/Documents/robotics-data/rlbench_data',headless=True)
# todo : Ensure Strict Variation Setting. 
# Set image_paths_output=False when loading dataset from file. 
demos = curr_env.get_demos(3,live_demos=False,image_paths_output=False) 
agent = GripperPoseRLAgent() 
num_episodes = 1000

for i in range(num_episodes):
    replay_buffer = curr_env.run_rl_episode(agent)
    agent.update(replay_buffer)