from metaflow import FlowSpec, step,retry


class FinalData():
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.agent_name = None
        self.loss = None
        self.simulation_analytics = None
        self.total_data_size = 0
    
    def __str__(self):
        num_convergence_metrics = len(self.simulation_analytics['convergence_metrics'])
        percent_converge = (len(self.simulation_analytics['convergence_metrics']) / self.simulation_analytics['total_epochs_allowed'])*100
        x = '''
        Agent Name : {agent_name}

        Simulation Results 

            Total Number of Episodes : {total_episodes}

            Steps Per Episode : {steps}

            Number Of Converged Cases : {num_convergence_metrics}

            %  Cases That Converged : {percent_converge}
        '''.format(agent_name=self.agent_name,\
            total_episodes=str(self.simulation_analytics['total_epochs_allowed']),\
            steps=str(self.simulation_analytics['max_steps_per_episode']),\
            num_convergence_metrics=str(num_convergence_metrics), \
            percent_converge=str(percent_converge)
            )
        return x

class TrainingSimulatorFlow(FlowSpec):

    @step
    def start(self):
        print("Importing data in this step")
        self.num_demos = 350
        self.num_epochs = 200 # Training epochs
        self.episode_length=50
        self.num_episodes=200 # Simulated Testing Epochs.
        self.agent_modules = [{
            'module_name':'models.SmartImmitationAgent',
            'agent_name':'SimpleImmitationLearningAgent'
        },{
            'module_name':'models.ImmitationMutant',
            'agent_name': 'ImmitationLearningMutantAgent'
        }]
        self.next(self.train,foreach='agent_modules')

    @retry(times=4)
    @step
    def train(self):
        # todo : Create an image for training 
        from SimulationEnvironment.Environment import ReachTargetSimulationEnv
        import importlib
        agent_module = importlib.import_module(self.input['module_name'])
        agent = getattr(agent_module,self.input['agent_name'])()

        curr_env = ReachTargetSimulationEnv(dataset_root='/tmp/rlbench_data')
        # Set image_paths_output=True when loading dataset from file if images also dont need to be loaded for dataset
        demos = curr_env.get_demos(self.num_demos,live_demos=False,image_paths_output=True) 
        # agent.load_model('SavedModels/ImmitationLearningConvolvingMutantAgent-2020-04-05-04-18.pt')
        agent.injest_demonstrations(demos)   
        loss = agent.train_agent(self.num_epochs)
        self.loss = loss
        self.total_data_size = agent.total_train_size
        self.agent_name = agent.__class__.__name__
        self.model = agent.neural_network.state_dict()
        self.optimizer = agent.optimizer.state_dict()
        self.next(self.simulate)
    
    @retry(times=4)
    @step
    def simulate(self):
        from SimulationEnvironment.Environment import ReachTargetSimulationEnv
        import importlib
        agent_module = importlib.import_module(self.input['module_name'])
        agent = getattr(agent_module,self.input['agent_name'])()

        curr_env = ReachTargetSimulationEnv(headless=True,episode_length=self.episode_length,num_episodes=self.num_episodes)
        agent.load_model_from_object(self.model)
        simulation_analytics = curr_env.run_trained_agent(agent)
        self.simulation_analytics = simulation_analytics
        self.next(self.join)

    @step
    def join(self,inputs):
        final_data = []
        for task_data in inputs:
            data = FinalData()
            data.model = task_data.model
            data.optimizer = task_data.optimizer
            data.agent_name = task_data.agent_name
            data.loss = task_data.loss
            data.simulation_analytics = task_data.simulation_analytics
            data.total_data_size = task_data.total_data_size
            final_data.append(data)
        
        self.final_data = final_data
        self.next(self.end)


    @step
    def end(self):
        print("Done Computation")

if __name__ == '__main__':
    TrainingSimulatorFlow()