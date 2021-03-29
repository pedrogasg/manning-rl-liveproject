from experiments import Experiment
from environments import WindyGridWorld
from policies import MCPolicyFV, MCPolicyES

class WindyWorldExperiment(Experiment):
    def run(self):
        grid = WindyGridWorld(6,6, wind=[0, 0, 1, 2, 1, 0])
        GAMMA = 0.9

        #policy = MCPolicyFV(grid, GAMMA)
        #times = 500
        times = 50000
        policy = MCPolicyES(grid, GAMMA)
        for i in range(times):
            if i % (times / 10) == 0:
                print('Starting episode', i)
            policy.episode()

        policy.print()

