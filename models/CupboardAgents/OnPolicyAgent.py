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
    
    def __init__(self,num_states,num_actions,hidden_dims=200):
        super(FullyConnectedPolicyEstimator, self).__init__()
        self.fc1 = nn.Linear(num_states, hidden_dims)
        self.fc3 = nn.Linear(hidden_dims, num_actions)

    # x is input to the network.
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc3(x)
        return x


class ConvolutionalPolicyEstimator(nn.Module):

    def __init__(self,num_actions,hidden_dims=50):
        super(ConvolutionalPolicyEstimator, self).__init__()
        # Image will be 128 * 128
        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc1 = nn.Linear(4*32*32, hidden_dims) # Input Dims Related to Op Dims of cnn_layer
        self.fc2 = nn.Linear(hidden_dims, hidden_dims)
        self.fc3 = nn.Linear(hidden_dims, num_actions)

    # x is input to the network.
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class GripperOpenEstimator(nn.Module):
    def __init__(self,image_impact_dims=2,ee_impact_dims=4,hidden_dims=30):
        super(GripperOpenEstimator,self).__init__()
        # Wrist Camera NN
        self.wrist_camera_rgb_convolution_NN = ConvolutionalPolicyEstimator(image_impact_dims) # Hypothesis : Output of this network will 
        # EE Pose NN
        self.end_effector_pose_NN = FullyConnectedPolicyEstimator(7,ee_impact_dims)

        self.fc1 = nn.Linear(ee_impact_dims+image_impact_dims,hidden_dims)
        self.fc3 = nn.Linear(hidden_dims, 1)

    def forward(self,EE_Pose,wrist_rgb):
        wrist_op = self.wrist_camera_rgb_convolution_NN(wrist_rgb)
        ee_op = self.end_effector_pose_NN(EE_Pose)
        stacked_tensor = torch.cat((wrist_op,ee_op),1)
        x = F.relu(self.fc1(stacked_tensor))
        x = self.fc3(x) # Todo : Check for softmax . 
        return x


class TargetPositionPolicy(nn.Module):

    def __init__(self,image_impact_dims=2,pred_hidden_dims=50):
        self.l_rgb_conversion = ConvolutionalPolicyEstimator(image_impact_dims)
        self.r_rgb_conversion = ConvolutionalPolicyEstimator(image_impact_dims)
        self.pre_pred_layer = nn.Linear(image_impact_dims*2+1,pred_hidden_dims) # Because of gripper open 
        self.output_layer = nn.Linear(pred_hidden_dims,3) # For X Y Z 

    def forward(self,left_rgb,right_rgb,gripper_open):
        left_rgb_op = self.l_rgb_conversion(left_rgb)
        right_rgb_op = self.r_rgb_conversion(right_rgb)
        stacked_tensor = torch.cat((left_rgb_op,right_rgb_op,gripper_open),1)
        x = F.relu(self.pre_pred_layer(stacked_tensor))
        x = self.output_layer(x)
        return x


class ActionModeEstimator(nn.Module):
    
    def __init__(self,action_dims=7,hidden_pre_pred_dims=30,joint_pos_policy_hidden=20):
        super(ModularPolicyEstimator, self).__init__()
        modular_policy_op_dims = 10
        # Define Modular Policy dimes
        self.joint_pos_policy = FullyConnectedPolicyEstimator(7,modular_policy_op_dims,hidden_dims=joint_pos_policy_hidden)
        # Define connecting Fc Linear layers.
        input_tensor_dims = modular_policy_op_dims + 3 + 1
        self.pre_pred_layer = nn.Linear(input_tensor_dims, hidden_pre_pred_dims)
        self.output_layer = nn.Linear(hidden_pre_pred_dims,action_dims)

    def forward(self,joint_pos,target_pos,gripper_open):
        joint_pos_op = self.joint_pos_policy(joint_pos)
        # option 1
        # stacked_tensor : shape (batch_size,)
        # Output is combined into a single tensor.
        stacked_tensor = torch.cat((joint_pos_op,target_pos,gripper_open),1)
        op = F.relu(self.fc1(stacked_tensor))
        op = self.fc3(op)
        return op




class ModularPolicyEstimator(nn.Module):
    def __init__(self):
        super(ModularPolicyEstimator,self).__init__()
        self.gripper_open_estimator = GripperOpenEstimator()
        self.action_mode_estimator = ActionModeEstimator()
        self.target_position_estimator = TargetPositionPolicy()
    
    def forward(self,EE_Pose,wrist_rgb,left_rgb,right_rgb,joint_positions):
        gripper_open = self.gripper_open_estimator(EE_Pose,wrist_rgb)
        target_position = self.target_position_estimator(left_rgb,right_rgb,gripper_open)
        pred_action = self.action_mode_estimator(joint_positions,target_position,gripper_open)        
        return gripper_open,pred_action,target_position




class OnPolicyAgent(TorchAgent):
    """
    OnPolicyAgent
    -----------------------

    
    """
    def __init__(self,learning_rate = 0.01,batch_size=64,collect_gradients=False):
        super(TorchAgent,self).__init__(collect_gradients=collect_gradients)
        self.learning_rate = learning_rate
        # action should contain 1 extra value for gripper open close state
        self.neural_network = ModularPolicyEstimator()
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
        joint_pos_arr = np.array([getattr(observation,'joint_positions') for episode in demos for observation in episode])
        # todo : Calculate Target pos Arra on basis  of the policy vector. 
        left_shoulder_rgb = np.array([getattr(observation,'left_shoulder_rgb') for episode in demos for observation in episode])
        right_shoulder_rgb = np.array([getattr(observation,'right_shoulder_rgb') for episode in demos for observation in episode])
        wrist_rgb = np.array([getattr(observation,'wrist_rgb') for episode in demos for observation in episode])
        



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