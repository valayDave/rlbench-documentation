# Import Absolutes deps
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from rlbench.backend.observation import Observation
from typing import List
import numpy as np
from torch.utils.data.dataset import Dataset
# Import Relative deps
import sys
sys.path.append('..')
from models.CupboardAgents.DDPG.ddpg import DDPG
from models.Agent import TorchRLAgent
import logger
 

class GripperPoseRLAgent(TorchRLAgent):
    """
    GripperPoseRLAgent
    -----------------------
    Algo Of Choice : https://spinningup.openai.com/en/latest/algorithms/ddpg.html

    Why DDPG : 
    - Our choice of input and output makes it ideal as
        1. GripperPoseRLAgent's input 
            1. `object_pose`
        2. Output is : `EE_pose` : This is the final pose. which is CONTINOUS but it can even be seen as deterministic. Our output will be the same 
        
        So as its a continous actionspace one can try and use DDPG 
        https://math.stackexchange.com/questions/3179912/policy-gradient-reinforcement-learning-for-continuous-state-and-action-space
        https://ai.stackexchange.com/questions/4085/how-can-policy-gradients-be-applied-in-the-case-of-multiple-continuous-actions

    TODO : FIX DDPG Action PREDS FOR Supporting allowed Quartienion Predictions
    TODO : FIX DDPG To Support ACTION SPACE FOR XYZ Based on Robot's Workspace.
    """
    def __init__(self,learning_rate = 0.001,batch_size=64,collect_gradients=False,warmup=1000):
        super(GripperPoseRLAgent,self).__init__(collect_gradients=collect_gradients,warmup=warmup)
        self.learning_rate = learning_rate
        # action should contain 1 extra value for gripper open close state
        self.neural_network = DDPG(7,8)
        self.agent_name ="DDPG__AGENT"
        self.logger = logger.create_logger(self.agent_name)
        self.logger.propagate = 0
        self.input_state = 'joint_positions'
        self.output_action = 'joint_velocities'
        self.data_loader = None
        self.dataset = None
        self.batch_size =batch_size
        self.desired_obj = 'chocolate_jello_grasp_point'
        self.print_every = 40
   
    
    def get_information_vector(self,demonstration_episode:List[Observation]):
        return demonstration_episode[0].object_poses[self.desired_obj]

    def observe(self,state_t1:List[Observation],action_t,reward_t:int,done:bool):
        state_t1 = self.get_information_vector(state_t1)
        self.neural_network.observe(reward_t,state_t1,done)
    
    def update(self):
        self.neural_network.update_policy()

    def act(self,state:List[Observation],timestep=0):
        """
        ACTION PREDICTION : ABS_EE_POSE_PLAN
        """
        # agent pick action ...
        if timestep <= self.warmup:
            action = self.neural_network.random_action()
        else:
            state = self.get_information_vector(state)
            action = self.neural_network.select_action(state)
        return action

