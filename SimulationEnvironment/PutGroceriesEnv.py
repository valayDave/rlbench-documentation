from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.backend.observation import Observation
from rlbench.tasks import PutGroceriesInCupboard
from typing import List
from quaternion import from_rotation_matrix, quaternion
import scipy as sp
import numpy as np
from gym import spaces
from .Environment import SimulationEnvionment
from .Environment import image_types,DEFAULT_ACTION_MODE,ArmActionMode

import sys
sys.path.append('..')
from models.Agent import LearningAgent,RLAgent
import logger



class PutGroceriesEnvironment(SimulationEnvionment):
    """
    Inherits the `SimulationEnvironment` class. 
    This environment is specially ment for running traing agent for PutGroceriesInCupboard Task. 
    This can be inherited for different ways of doing learning. 
    
    :param num_episodes : Get the total Epochs needed for the Simulation
    """
    def __init__(self, action_mode=DEFAULT_ACTION_MODE, headless=True,num_episodes = 120,episode_length = 40,dataset_root=''):
        super(PutGroceriesEnvironment,self).__init__(action_mode=action_mode, task=PutGroceriesInCupboard, headless=headless,dataset_root=dataset_root)
        self.num_episodes = num_episodes
        self.episode_length = episode_length
        self.logger = logger.create_logger(__class__.__name__)
        self.logger.propagate = 0


    def _get_state(self, obs:Observation,check_images=True):
        # _get_state function is present so that some alterations can be made to observations so that
        # dimensionality management is handled from lower level. 
        if not check_images: # This is set so that image loading can be avoided
            return obs

        for state_type in image_types:    # changing axis of images in `Observation`
            image = getattr(obs, state_type)
            if image is None:
                continue
            if len(image.shape) == 2:
                # Depth and Mask can be single channel.Hence we Reshape the image from (width,height) -> (width,height,1)
                image = image.reshape(*image.shape,1)
            # self.logger.info("Shape of : %s Before Move Axis %s" % (state_type,str(image.shape)))
            image=np.moveaxis(image, 2, 0)  # change (H, W, C) to (C, H, W) for torch
            # self.logger.info("After Moveaxis :: %s" % str(image.shape))
            setattr(obs,state_type,image)

        return obs

    def run_trained_agent(self,agent:LearningAgent):
        simulation_analytics = {
            'total_epochs_allowed':self.num_episodes,
            'max_steps_per_episode': self.episode_length,
            'convergence_metrics':[]
        }
        rest_step_counter = 0
        total_steps = 0
        for i in range(self.num_episodes):
            rest_step_counter=0
            self.logger.info('Reset Episode %d'% i)
            obs,descriptions = self.task_reset()
            self.logger.info(descriptions)

            for _ in  range(self.episode_length): # Iterate for each timestep in Episode length
                action = agent.predict_action([obs])
                selected_action = action
                # print(selected_action,"selected_action")
                obs, reward, terminate = self.step(selected_action)
                rest_step_counter+=1
                if reward == 1:
                    self.logger.info("Reward Of 1 Achieved. Task Completed By Agent In steps : %d"%rest_step_counter)
                    simulation_analytics['convergence_metrics'].append({
                        'steps_to_convergence': rest_step_counter,
                        'epoch_num':i
                    })

                if terminate:
                    break
                
        self.shutdown()
        return simulation_analytics

    def task_reset(self):
        descriptions, obs = self.task.reset()
        obs = self._get_state(obs)
        return obs,descriptions


    def get_demos(self,num_demos,live_demos=True,image_paths_output=False):
        """
        :param: live_demos : If live_demos=True,
        :param: image_paths_output : Useful set to True when used with live_demos=False. If set True then the dataset loaded from FS will not load the images but will load the paths to the images. 
        """
        self.logger.info("Creating Demos")
        demos = self.task.get_demos(num_demos, live_demos=live_demos,image_paths=image_paths_output)  # -> List[List[Observation]]
        self.logger.info("Created Demos")
        demos = np.array(demos).flatten()
        self.shutdown()
        new_demos = []
        if live_demos:
            check_images = True
        else:
            if image_paths_output:
                check_images = False
            else:
                check_images=True
        for episode in demos:
            new_episode = []
            for step in episode:
                # Only transform images to in `Observation` object if its a live_demo or when image_out_path=False with live_demo=False
                new_episode.append(self._get_state(step,check_images=check_images)) 
            new_demos.append(new_episode)
        return new_demos


#### POSE SETUP METHODS FOR THIS CODE ########
class NoisyObjectPoseSensor:
    """
    Credits : Kevin(Rohan's Friend)
    """
    def __init__(self, env):
        self._env = env
        self._pos_scale = [0.005] * 3
        self._rot_scale = [0.01] * 3
    def get_poses(self):
        objs = self._env._scene._active_task.get_base().get_objects_in_tree(exclude_base=True, first_generation_only=False)
        obj_poses = {}
        for obj in objs:
            name = obj.get_name()
            pose = obj.get_pose()
            pos, quat_wxyz = sample_normal_pose(self._pos_scale, self._rot_scale)
            gt_quat_wxyz = quaternion(pose[6], pose[3], pose[4], pose[5])
            perturbed_quat_wxyz = quat_wxyz * gt_quat_wxyz
            pose[:3] += pos
            pose[3:] = [perturbed_quat_wxyz.x, perturbed_quat_wxyz.y, perturbed_quat_wxyz.z, perturbed_quat_wxyz.w]
            obj_poses[name] = pose
        return obj_poses

def skew(x):
    return np.array([[0, -x[2], x[1]],
                    [x[2], 0, -x[0]],
                    [-x[1], x[0], 0]])
def sample_normal_pose(pos_scale, rot_scale):
    '''
    Samples a 6D pose from a zero-mean isotropic normal distribution
    '''
    pos = np.random.normal(scale=pos_scale)
    eps = skew(np.random.normal(scale=rot_scale))
    R = sp.linalg.expm(eps)
    quat_wxyz = from_rotation_matrix(R)
    return pos, quat_wxyz
####################################################

class ReplayBuffer():
    
    def __init__(self):
        self.observations = []
        self.rewards = []
        self.actions = []
        self.total_reward = 0

    def store(self,observation:Observation,action,reward:int):
        self.observation.append(observation)
        self.actions.append(action)
        self.observation.append(reward)
"""
1. Imitation learning baseline :
    1. Mofify RLBench to support Sensor reads at output. 

    2.Predict :: EEPose 
    ACTION MODE : DELTA_EE_POSE 
"""    

class PutGrocceriesRLGraspingEnvironment(PutGroceriesEnvironment):
    def __init__(self, action_mode=ArmActionMode.DELTA_EE_POSE, headless=True, num_episodes=120, episode_length=40, dataset_root=''):
        super().__init__(action_mode=action_mode, headless=headless, num_episodes=num_episodes, episode_length=episode_length, dataset_root=dataset_root)
        self.obj_pose_sensor = NoisyObjectPoseSensor(self.env)
        # TODO : Set Reward Based Proximity Sensors IN our Environment.

    def get_object_poses(self):
        return self.obj_pose_sensor.get_poses()

    def reward_function(self,state:Observation,action,rlbench_reward:int):
        """
        reward_function : Reward function for non Sparse Rewards. 
        Parameters
        ----------
        rlbench_reward :this specifies the reward from RLBench . 
                        When 1 It means we completed the Task. 
        state : Actual State Obs.
        action: Action taken  by the agent in case needed for reward shaping

        """
        # todo : Shape Reward from env here. 
        # TODO : SHAPE REWAED HERE> By Readings from Proximity Sensors. 
        if rlbench_reward ==1:
            return 1            
        return 0


    def _get_state(self, obs:Observation,check_images=True):
        # _get_state function is present so that some alterations can be made to observations so that
        # dimensionality management is handled from lower level. 
        if not check_images: # This is set so that image loading can be avoided
            return self.set_object_pose(obs)

        for state_type in image_types:    # changing axis of images in `Observation`
            image = getattr(obs, state_type)
            if image is None:
                continue
            if len(image.shape) == 2:
                # Depth and Mask can be single channel.Hence we Reshape the image from (width,height) -> (width,height,1)
                image = image.reshape(*image.shape,1)
            # self.logger.info("Shape of : %s Before Move Axis %s" % (state_type,str(image.shape)))
            image=np.moveaxis(image, 2, 0)  # change (H, W, C) to (C, H, W) for torch
            # self.logger.info("After Moveaxis :: %s" % str(image.shape))
            setattr(obs,state_type,image)

        return self.set_object_pose(obs)
    
    def set_object_pose(self,obs):
        # Add Object pose as a part of OBS.
        attr_name = 'object_poses'
        setattr(obs,attr_name,self.get_object_poses)
        return obs

    def step(self, action):
        obs_, reward, terminate = self.task.step(action)  # reward in original rlbench is binary for success or not
        state_obs = self._get_state(obs_)
        shaped_reward = self.reward_function(state_obs,action,reward)
        return state_obs, shaped_reward, terminate


    def run_rl_episode(self,agent:RLAgent) -> ReplayBuffer:
        """
        This function should be used under the following assumption 
        
        1.  You care about total reward attained For the epsiode instead of the rewards at time steps.
            Hence the `predict_action` function is used here which is not ment for any gradient flow.
            ReplayBuffer is returned which hold whole episode and rewards at each Timestep.
        """        
        # obs_traj = 
        replay_buffer = ReplayBuffer()
        obs,descriptions = self.task_reset()
        prev_obs = obs
        for step_counter in range(self.episode_length): # Iterate for each timestep in Episode length
            action = agent.predict_action([obs],time_step=step_counter) # Provide state s_t to agent.
            selected_action = action
            obs, reward, terminate = self.step(selected_action)
            replay_buffer.store(prev_obs,action,reward) # s_t,a_t,reward_t : This should also be thought out.
            replay_buffer.total_reward+=reward
            prev_obs = obs
            if terminate:
                self.logger.info("Terminating!!")
                break

        return replay_buffer