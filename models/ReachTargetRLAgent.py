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
# Import Relative deps
import sys
sys.path.append('..')
from models.Agent import TorchRLAgent
import logger
from models.CupboardAgents.DDPG.ddpg import DDPG,DDPGArgs
 


class ReachTargetRLAgent(TorchRLAgent):
    """
    ReachTargetRLAgent
    -----------------------
    Algo Of Choice : https://spinningup.openai.com/en/latest/algorithms/ddpg.html
    
    ABS_JOINT_VELOCITY

    Why DDPG : 
   
        So as its a continous actionspace one can try and use DDPG 
        https://math.stackexchange.com/questions/3179912/policy-gradient-reinforcement-learning-for-continuous-state-and-action-space
        https://ai.stackexchange.com/questions/4085/how-can-policy-gradients-be-applied-in-the-case-of-multiple-continuous-actions

    """
    def __init__(self,learning_rate = 0.001,batch_size=10,collect_gradients=False,warmup=50):
        super(ReachTargetRLAgent,self).__init__(collect_gradients=collect_gradients,warmup=warmup)
        self.learning_rate = learning_rate
        # action should contain 1 extra value for gripper open close state
        self.neural_network = DDPG(13,8) # 1 DDPG Setup with Different Predictors. 
        self.agent_name ="DDPG__AGENT"
        self.logger = logger.create_logger(self.agent_name)
        self.logger.propagate = 0
        self.input_state = 'joint_positions'
        self.output_action = 'joint_velocities'
        self.data_loader = None
        self.dataset = None
        self.batch_size =batch_size
        self.print_every = 40
   
    
    def get_information_vector(self,demonstration_episode:List[Observation]):
        joint_pos_arr = np.array([getattr(observation,'joint_positions') for observation in demonstration_episode])
        target_pos_arr = np.array([getattr(observation,'task_low_dim_state') for observation in demonstration_episode])
        gripper_pose = np.array([getattr(observation,'gripper_pose')[:3] for observation in demonstration_episode])
        final_vector = np.concatenate((joint_pos_arr,
                        target_pos_arr,
                        gripper_pose),axis=1)
        return final_vector

    def observe(self,state_t1:List[Observation],action_t,reward_t:int,done:bool):
        """
        State s_t1 can be None because of errors thrown by policy. 
        """
        state_t1 = None if state_t1[0] is None else self.get_information_vector(state_t1)
        self.neural_network.observe(reward_t,state_t1,done)
    
    def update(self):
        self.neural_network.update_policy()

    def reset(self,state:List[Observation]):
        self.neural_network.reset(self.get_information_vector(state))

    def act(self,state:List[Observation],timestep=0):
        """
        ACTION PREDICTION : ABS_JOINT_VELOCITY
        """
        # agent pick action ...
        if timestep <= self.warmup:
            action = self.neural_network.random_action()
        else:
            state = self.get_information_vector(state)
            action = self.neural_network.select_action(state)[0] # 8 Dim Vector
        action = list(action)
        return action