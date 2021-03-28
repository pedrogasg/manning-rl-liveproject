class BaseAgent(object):
    def __init__(self):        
        self.memory = []
        self.initVariables() 

    def chooseAction(self, state):           
        pass

    def print(self):
        pass

    def updateMemory(self, state, reward):
        self.memory.append((state, reward))

    def update(self):
        pass