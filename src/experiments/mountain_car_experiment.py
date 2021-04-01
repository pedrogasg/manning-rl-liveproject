import gym
import numpy as np
import matplotlib.pyplot as plt
from experiments import Experiment
from policies import MCAPolicy, TDPolicy

class MountainCarExperiment(Experiment):

    def run(self):
        env = gym.make('MountainCar-v0')
        
        numEpisodes = 20000

        GAMMA = 1.0

        nearExit = np.zeros((3, int(numEpisodes/1000)))
        leftSide = np.zeros((3, int(numEpisodes/1000)))
        x = np.arange(nearExit.shape[1])

        for k, ALPHA in enumerate([1e-1, 1e-2, 1e-3]):
            #policy = MCAPolicy(env, ALPHA, GAMMA)
            policy = TDPolicy(env, ALPHA, GAMMA)

            for i in range(numEpisodes):

                if i % 1000 == 0:
                    print('start episode', i)
                    idx = i // 1000
                    state = policy.aggregateState((0.43, 0.054))
                    nearExit[k][idx] = policy.calculateV(state)        
                    state = policy.aggregateState((-1.1, 0.001))
                    leftSide[k][idx] = policy.calculateV(state)
                    #policy.dt += 0.1
                if i % 100 == 0:
                    policy.dt += 10
                policy.train()

        plt.subplot(221)
        plt.plot(x, nearExit[0], 'r--')
        plt.plot(x, nearExit[1], 'g--')
        plt.plot(x, nearExit[2], 'b--')
        plt.title('near exit, moving right')
        plt.subplot(222)    
        plt.plot(x, leftSide[0], 'r--')
        plt.plot(x, leftSide[1], 'g--')
        plt.plot(x, leftSide[2], 'b--')
        plt.title('left side, moving right')
        plt.legend(('alpha = 0.1', 'alpha = 0.01', 'alpha = 0.001'))
        plt.show()