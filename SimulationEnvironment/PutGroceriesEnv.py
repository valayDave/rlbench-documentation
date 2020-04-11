from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.backend.observation import Observation
from rlbench.tasks import PutGroceriesInCupboard
from typing import List
import numpy as np
from gym import spaces
from .Environment import SimulationEnvionment
from .Environment import image_types,DEFAULT_ACTION_MODE

import sys
sys.path.append('..')
from models.Agent import LearningAgent
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