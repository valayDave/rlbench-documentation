from metaflow import FlowSpec, step,retry
import json

class FinalData():
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.agent_name = None
        self.loss = None
        self.simulation_analytics = None
        self.total_data_size = 0
        self.model_args = {}
        self.gradients = {
            'max':[],
            'avg':[],
            'layer':[]
        }
    
    def __str__(self):
        num_convergence_metrics = len(self.simulation_analytics['convergence_metrics'])
        percent_converge = (len(self.simulation_analytics['convergence_metrics']) / self.simulation_analytics['total_epochs_allowed'])*100
        data_size = 0
        if hasattr(self,'total_data_size'):
            data_size = self.total_data_size
       
        model_args = {}
        if hasattr(self,'model_args'):
            model_args = self.model_args
        
        collected_grads = "No"
        if hasattr(self,'gradients'):
            if len(self.gradients['avg']) > 0:
                collected_grads = 'Yes'

        x = '''
        Agent Name : {agent_name}

        Total Training Data Size : {data_size}

        Model Arguements : {model_args}

        Collected Gradients : {grad_collect}

        Simulation Results 

            Total Number of Episodes : {total_episodes}

            Steps Per Episode : {steps}

            Number Of Converged Cases : {num_convergence_metrics}

            %  Cases That Converged : {percent_converge}
        '''.format(agent_name=self.agent_name,\
            total_episodes=str(self.simulation_analytics['total_epochs_allowed']),\
            steps=str(self.simulation_analytics['max_steps_per_episode']),\
            num_convergence_metrics=str(num_convergence_metrics), \
            percent_converge=str(percent_converge), \
            model_args=json.dumps(model_args), \
            data_size="NO DATA" if data_size is 0 else str(data_size),
            grad_collect=collected_grads
            )
        return x

class TrainingSimulatorFlow(FlowSpec):

    @step
    def start(self):
        print("Importing data in this step")
        self.num_demos=49
        self.num_epochs=5 # Training epochs
        self.episode_length=100
        self.num_episodes=2 # Simulated Testing Epochs.
        self.variation_number = 0     
        self.collect_gradients = True                                                                                                                                                                           
        self.agent_modules = [{
            'module_name':'models.SmartImmitationAgent',
            'agent_name':'SimpleImmitationLearningAgent',
            'args':{'num_layers':4},
            'reporting_name':'SimpleImmitationLearningAgent__4'
        },
        {
            'module_name':'models.ImmitationLearning',
            'agent_name':'ImmitationLearningAgent',
            'args':{},
            'reporting_name':'Idiot_ImmitationLearningAgent'
        },
        {
            'module_name':'models.ImmitationMutationConv',
            'agent_name':'ImmitationLearningConvolvingMutantAgent',
            'args':{},
            'reporting_name':'ImmitationLearningConvolvingMutantAgent'
        },
        {
            'module_name':'models.SmartImmitationAgent',
            'agent_name':'SimpleImmitationLearningAgent',
            'args':{'num_layers':1},
            'reporting_name':'SimpleImmitationLearningAgent__1'
        },
        {
            'module_name':'models.SmartImmitationAgent',
            'agent_name':'SimpleImmitationLearningAgent',
            'args':{'num_layers':2},
            'reporting_name':'SimpleImmitationLearningAgent__2'
        },
        {
            'module_name':'models.ImmitationMutant',
            'agent_name': 'ImmitationLearningMutantAgent',
            'args':{},
            'reporting_name':'ImmitationLearningMutantAgent'
        }]
        self.next(self.train,foreach='agent_modules')

    @retry(times=4)
    @step
    def train(self):
        # todo : Create an image for training 
        from SimulationEnvironment.Environment import ReachTargetSimulationEnv
        import importlib
        agent_module = importlib.import_module(self.input['module_name'])
        agent = getattr(agent_module,self.input['agent_name'])(**self.input['args'],collect_gradients=self.collect_gradients)

        curr_env = ReachTargetSimulationEnv(dataset_root='/tmp/rlbench_data')
        curr_env.task._variation_number = self.variation_number
        # Set image_paths_output=True when loading dataset from file if images also dont need to be loaded for dataset
        demos = curr_env.get_demos(self.num_demos,live_demos=False,image_paths_output=False) 
        # agent.load_model('SavedModels/ImmitationLearningConvolvingMutantAgent-2020-04-05-04-18.pt')
        agent.injest_demonstrations(demos)   
        loss = agent.train_agent(self.num_epochs)
        self.loss = loss
        self.total_data_size = agent.total_train_size
        self.agent_name = self.input['reporting_name']
        self.model = agent.neural_network.state_dict()
        self.model_args = self.input['args']
        self.optimizer = agent.optimizer.state_dict()
        self.gradients = agent.gradients 
        self.next(self.simulate)
    
    @retry(times=4)
    @step
    def simulate(self):
        from SimulationEnvironment.Environment import ReachTargetSimulationEnv
        import importlib
        agent_module = importlib.import_module(self.input['module_name'])
        agent = getattr(agent_module,self.input['agent_name'])(**self.input['args'])

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
            data.model_args = task_data.model_args
            data.gradients = task_data.gradients 
            final_data.append(data)
        
        self.final_data = final_data
        self.next(self.end)


    @step
    def end(self):
        print("Done Computation")

if __name__ == '__main__':
    TrainingSimulatorFlow()