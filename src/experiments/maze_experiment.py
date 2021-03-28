import numpy as np
from environments import Maze
from agents import Agent
import matplotlib.pyplot as plt

from experiments import Experiment 

class MazeExperiment(Experiment):

    def run(self):
        maze = Maze()
        robot = Agent(maze, alpha=0.1, randomFactor=0.25)
        moveHistory = []
        for i in range(5000):
            if i % 1000 == 0:
                print(i)
            while not maze.isGameOver():
                state, _ = maze.getStateAndReward()
                action = robot.chooseAction(state, maze.allowedStates[state])
                maze.updateMaze(action)
                state, reward = maze.getStateAndReward()
                robot.updateStateHistory(state, reward)
                if maze.steps > 1000:
                    maze.robotPosition = (5,5)
            robot.learn()
            moveHistory.append(maze.steps)
            maze = Maze()

        maze = Maze()
        robot = Agent(maze, alpha=0.5, randomFactor=0.25)
        moveHistory2 = []
        for i in range(5000):
            if i % 1000 == 0:
                print(i)
            while not maze.isGameOver():
                state, _ = maze.getStateAndReward()
                action = robot.chooseAction(state, maze.allowedStates[state])
                maze.updateMaze(action)
                state, reward = maze.getStateAndReward()
                robot.updateStateHistory(state, reward)
                if maze.steps > 1000:
                    maze.robotPosition = (5,5)            
            robot.learn()
            moveHistory2.append(maze.steps)
            maze = Maze()

        plt.subplot(211)
        plt.semilogy(moveHistory, 'b--')
        plt.legend(['alpha=0.1'])
        plt.subplot(212)
        plt.semilogy(moveHistory2, 'r--')
        plt.legend(['alpha=0.99'])
        plt.show()