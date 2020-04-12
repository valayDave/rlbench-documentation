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
            1. `start_state_gripper_pose`
            2. `object_pose`
        2. Output is : `EE_pose` : This is the final pose. which is CONTINOUS but it can even be seen as deterministic. Our output will be the same 
        
        So as its a continous actionspace one can try and use DDPG 
        https://math.stackexchange.com/questions/3179912/policy-gradient-reinforcement-learning-for-continuous-state-and-action-space
        https://ai.stackexchange.com/questions/4085/how-can-policy-gradients-be-applied-in-the-case-of-multiple-continuous-actions

    
    """
    def __init__(self,learning_rate = 0.01,batch_size=64,collect_gradients=False):
        super(GripperPoseRLAgent,self).__init__(collect_gradients=collect_gradients)
        self.learning_rate = learning_rate
        # action should contain 1 extra value for gripper open close state
        self.neural_network = None
        self.optimizer = None
        self.loss_function = None
        self.training_data = None
        self.logger = logger.create_logger(__class__.__name__)
        self.logger.propagate = 0
        self.input_state = 'joint_positions'
        self.output_action = 'joint_velocities'
        self.data_loader = None
        self.dataset = None
        self.batch_size =batch_size
        self.desired_obj = 'chocolate_jello_grasp_point'
        self.object_pose = None
        self.gripper_start_pose = None
   
    
    def set_episode_constants(self,demonstration_episode:List[Observation]):
        self.object_pose = demonstration_episode[0].object_poses[self.desired_obj]
        self.gripper_start_pose = demonstration_episode[0].gripper_pose


    def predict_action(self, demonstration_episode:List[Observation],time_step=None) -> np.array:
        """
        ACTION PREDICTION : DELTA_EE_POSE
        """
        if time_step == 0:
            self.set_episode_constants(demonstration_episode)

        # TODO :: NN Input and Output
        nn_op = None
        
        self.act_rel(nn_op)
        return op # Because there is only one action as output. 

    
    def act_rel(self,gripper,des):
        step = des - gripper
        step /= 10
        quat = [0,0,0,1]
        # norm = np.linalg.norm(quat)
        # quat /= norm
        step[3:7] = quat
        # step[1:7] = 0
        # step[3:7] = [1/2,1/2,1/2,1/2]
        return list(step) + [1]
