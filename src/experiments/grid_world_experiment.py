from experiments import Experiment
from environments import GridWorld
from policies.policy_iterator import PolicyIteration
class GridWordlExperiment(Experiment):
    def run(self):
        grid = GridWorld(4,4)
        THETA = 10e-6
        GAMMA = 1.0
        
        policy = PolicyIteration(grid, GAMMA, THETA)
        
        # main loop for policy improvement
        
        while not policy.stable:
            policy.evaluatePolicy()
            policy.improvePolicy()       
        
        
        #for i in range(2):
            #policy.iterateValues()
        
        policy.print()