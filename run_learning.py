# import deep_learning_rl as Simulator  
from SimulationEnvironment.PutGroceriesEnv import PutGroceriesRLEnvironment
from models.CupboardAgents.GripperPoseAgent import GripperPoseRLAgent    
import time

#TODO: 
# how do we share a dataset root folder without file path conflict?
# 

# Set dataset_root to load from a folder and dataset will load from there. 
curr_env = PutGroceriesRLEnvironment(dataset_root='/home/valay/Documents/robotics-data/rlbench_data',headless=True)
# todo : Ensure Strict Variation Setting. 
# Set image_paths_output=False when loading dataset from file. 
# demos = curr_env.get_demos(3,live_demos=False,image_paths_output=False) 
agent = GripperPoseRLAgent() 
curr_env.train_rl_agent(agent)