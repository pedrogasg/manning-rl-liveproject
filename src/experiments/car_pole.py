from experiments import Experiment
import gym
import numpy as np
import matplotlib.pyplot as plt

from policies import SimplePolicy, SarSaPolicy, QLearningPolicy
from policies import DoubleQlearning


def plotRunningAverage(totalrewards):
    N = len(totalrewards)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(totalrewards[max(0, t-100):(t+1)])
    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()

class CarPoleExperiment(Experiment):
    def run(self):
        env = gym.make('CartPole-v0')
        ALPHA = 0.1
        GAMMA = 1.0
        EPS = 1.0
        #numGames = 1000
        #policy = SimplePolicy(env, ALPHA, GAMMA)
        #numGames = 50000
        #policy = SarSaPolicy(env, ALPHA, GAMMA, EPS, numGames)
        #policy = QLearningPolicy(env, ALPHA, GAMMA, EPS, numGames)
        numGames = 100000
        policy = DoubleQlearning(env, ALPHA, GAMMA, EPS, numGames)
        totalRewards = np.zeros(numGames)
        for i in range(numGames):
            if i % (numGames/10) == 0:
                print('starting game', i)
            reward = policy.train()
            totalRewards[i] = reward
        
        plotRunningAverage(totalRewards)
        