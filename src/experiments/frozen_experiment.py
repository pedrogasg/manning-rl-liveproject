from experiments.experiment import Experiment


import gym

from agents import FrozenAgent
import matplotlib.pyplot as plt 
from experiments import Experiment

class FrozenExperiment(Experiment):
    def run(self):
        env = gym.make('FrozenLake-v0')
        states = [x for x in range(16)]         
        robot = FrozenAgent(gamma=0.9, states=states, policy=None)

        episodeRewards = []
        rewards = 0
        for episode in range(1000):
            done = False
            observation = env.reset()        
            while not done:
                action = env.action_space.sample() # randomly sample actions in action space
                observation, reward, done, info = env.step(action)
                robot.updateMemory(observation, reward) 
                rewards += reward
            robot.update()
            episodeRewards.append(rewards)
        robot.print()
        plt.plot(episodeRewards)
        plt.show()
        print('\n------------------------\n')

        # an attempt at a reasonable policy
        # 0 = left, 1 = down, 2 = right, 3 = up
        directedPolicy = {
            0: 1,
            1: 2,
            2: 1,
            3: 0,
            4: 1,
            5: 1,
            6: 1,
            7: 1,
            8: 2,
            9: 1,
            10: 1,
            11: 1,
            12: 2,
            13: 2,
            14: 2
        }
        
        robot = FrozenAgent(gamma=0.9, states=states, policy=directedPolicy)
        episodeRewards = []
        rewards = 0
        for episode in range(1000):
            done = False
            observation = env.reset()        
            while not done:
                action = robot.chooseAction(observation)
                observation, reward, done, info = env.step(action)
                robot.updateMemory(observation, reward)  
                rewards += reward     
            robot.update()
            episodeRewards.append(rewards)
        robot.print()
        plt.plot(episodeRewards)
        plt.show()