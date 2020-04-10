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
from models.Agent import TorchAgent
import logger
 


class FullyConnectedPolicyEstimator(nn.Module):
    
    def __init__(self,num_states,num_actions):
        super(FullyConnectedPolicyEstimator, self).__init__()
        self.fc1 = nn.Linear(num_states, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, num_actions)

    # x is input to the network.
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class GripperOpenEstimator(nn.Module):
    def __init__(self):
        super(GripperOpenEstimator,self).__init__()
        self.wrist_camera_rgb_convolution_NN = None
        self.end_effector_pose_NN = None
        self.fc3 = nn.Linear(200, 1)



class ActionModeEstimator(nn.Module):
    
    def __init__(self,action_dims=7):
        super(ModularPolicyEstimator, self).__init__()
        modular_policy_op_dims = 10
        # Define Modular Policy dimes
        self.joint_pos_policy = FullyConnectedPolicyEstimator(7,modular_policy_op_dims)
        # Define connecting Fc Linear layers.
        self.fc1 = nn.Linear(20, 200)
        self.fc3 = nn.Linear(200,action_dims)

    def forward(self,joint_pos,target_pos):
        joint_pos_op = self.joint_pos_policy(joint_pos)
        target_pos_op = self.target_pos_policy(target_pos)
        
        # option 1
        # stacked_tensor : shape (batch_size,10+10)
        # Output is combined into a single tensor.
        stacked_tensor = torch.cat((joint_pos_op,target_pos_op),1)
        op = F.relu(self.fc1(stacked_tensor))
        op = self.fc3(op)

        return op


class ActionModeEstimator(nn.Module):
    def __init__(self):
        super(ActionModeEstimator,self).__init__()
        self.


class ModularPolicyEstimator(nn.Module):
    def __init__(self):
        super(ModularPolicyEstimator,self).__init__()
        self.gripper_open_estimator = None
        self.action_mode_estimator = None





class OnPolicyAgent(TorchAgent):
    """
    OnPolicyAgent
    -----------------------

    
    """
    def __init__(self,learning_rate = 0.01,batch_size=64,collect_gradients=False):
        super(TorchAgent,self).__init__(collect_gradients=collect_gradients)
        self.learning_rate = learning_rate
        # action should contain 1 extra value for gripper open close state
        self.neural_network = FullyConnectedPolicyEstimator(7,8)
        self.optimizer = optim.SGD(self.neural_network.parameters(), lr=learning_rate, momentum=0.9)
        self.loss_function = nn.SmoothL1Loss()
        self.training_data = None
        self.logger = logger.create_logger(__name__)
        self.logger.propagate = 0
        self.input_state = 'joint_positions'
        self.output_action = 'joint_velocities'
        self.data_loader = None
        self.dataset = None
        self.batch_size =batch_size

    def injest_demonstrations(self,demos:List[List[Observation]],**kwargs):
        joint_position_train_vector = torch.from_numpy(self.get_train_vectors(demos))
        self.total_train_size = len(joint_position_train_vector)
        # $ First Extract the output_action. Meaning the action that will control the kinematics of the robot. 
        ground_truth_velocities = np.array([getattr(observation,'joint_velocities') for episode in demos for observation in episode]) #
        # $ Create a matrix for gripper position vectors.                                                                                                                                                     
        ground_truth_gripper_positions = np.array([getattr(observation,'gripper_open') for episode in demos for observation in episode])
        # $ Final Ground truth Tensor will be [joint_velocities_0,...joint_velocities_6,gripper_open]
        ground_truth_gripper_positions = ground_truth_gripper_positions.reshape(len(ground_truth_gripper_positions),1)
        ground_truth = torch.from_numpy(np.concatenate((ground_truth_velocities,ground_truth_gripper_positions),axis=1))
        
        # demos[0][0].task_low_dim_state contains all target's coordinates
        self.logger.info("Creating Tensordata for Pytorch of Size : %s"%str(joint_position_train_vector.size()))
        self.dataset = torch.utils.data.TensorDataset(joint_position_train_vector, ground_truth)
        
        self.data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def get_train_vectors(self,demos:List[List[Observation]]):
        return np.array([getattr(observation,'joint_positions') for episode in demos for observation in episode])

    def train_agent(self,epochs:int):
        if not self.dataset:
            raise Exception("No Training Data Set to Train Agent. Please set Training Data using ImmitationLearningAgent.injest_demonstrations")
        
        self.logger.info("Starting Training of Agent ")
        self.neural_network.train()
        for epoch in range(epochs):
            running_loss = 0.0
            steps = 0
            for batch_idx, (data, target) in enumerate(self.data_loader):
                data, target = Variable(data), Variable(target)
                self.optimizer.zero_grad()
                network_pred = self.neural_network(data.float()) 
                loss = self.loss_function(network_pred,target.float())
                loss.backward()
                if self.collect_gradients:
                    self.set_gradients(self.neural_network.named_parameters())
                self.optimizer.step()
                running_loss += loss.item()
                steps+=1

            self.logger.info('[%d] loss: %.6f' % (epoch + 1, running_loss / (steps+1)))

    def predict_action(self, demonstration_episode:List[Observation],**kwargs) -> np.array:
        self.neural_network.eval()
        train_vectors = self.get_train_vectors([demonstration_episode])
        input_val = Variable(torch.from_numpy(train_vectors[0]))
        output = self.neural_network(input_val.float())
        return output.data.cpu().numpy()
        # return np.random.uniform(size=(len(batch), 7))