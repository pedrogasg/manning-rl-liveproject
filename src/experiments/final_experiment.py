import gym

import numpy as np

import matplotlib.pyplot as plt

from experiments import Experiment

from policies import FinalPolicy

class FinalExperiment(Experiment):
    def run(self):
        env = gym.make('MountainCar-v0')
        env._max_episode_steps = 1000
        GAMMA = 1
        EPSILON = 1.0

        numEpisodes = 500
        numRuns = 50
        epLengths = np.zeros((3, numEpisodes, numRuns))
        x = np.arange(epLengths.shape[1])

        for k, ALPHA in enumerate([0.01, 0.1, 0.2]):
            for j in range(numRuns):
                print('alpha', ALPHA, 'run ', j)
                policy = FinalPolicy(env, ALPHA, GAMMA, EPSILON)

                for i in range(numRuns):
                    if i % 100 == 0:
                        policy.dt += 1
                    epLengths[k][i][j] = policy.train()
                    if policy.EPSILON - 2 / numEpisodes > 0:
                        policy.EPSILON -= 2 / numEpisodes
                    else:
                        policy.EPSILON = 0

        averaged1 = np.mean(epLengths[0], axis=1)    
        averaged2 = np.mean(epLengths[1], axis=1)
        averaged3 = np.mean(epLengths[2], axis=1)

        plt.plot(averaged1, 'r--')
        plt.plot(averaged2, 'b--')
        plt.plot(averaged3, 'g--')

        plt.legend(('alpha = 0.01', 'alpha = 0.1', 'alpha = 0.2'))
        plt.show()