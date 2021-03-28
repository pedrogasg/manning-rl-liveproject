

import numpy as np
from matplotlib import pyplot as plt
from bandits import Bandit
from experiments import Experiment

class BanditExperiment(Experiment):
    @staticmethod
    def simulate(numArms, epsilon, numPulls, C, initialQ, mode):    
        rewardHistory = np.zeros(numPulls)
        for j in range(2000):
            rewards = [np.random.randn() for _ in range(numArms)]
            bandit = Bandit(numArms, rewards, epsilon, C, initialQ, mode)
            if j % 200 == 0:
                print(j)
            for i in range(numPulls):        
                reward = bandit.pull()
                bandit.updateMean(reward)
                rewardHistory[i] += reward

        average = rewardHistory / 2000
        return average

    def run(self):
        numActions = 5    
        run1 = BanditExperiment.simulate(numActions, epsilon=0.1, numPulls=1000, C=0, initialQ=0, mode='constant')
        run2 = BanditExperiment.simulate(numActions, epsilon=0.0, numPulls=1000, C=2, initialQ=10, mode='constant')
        plt.plot(run1, 'b--', run2, 'r--')
        plt.legend(['Realistic epsilon greedy', 'Optimistic pure greedy'])
        plt.show()

    