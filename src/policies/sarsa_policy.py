import numpy as np

class SarSaPolicy(object):

    def __init__(self, env, ALPHA, GAMMA, EPS, weightReduction):
        self.env = env
        self.GAMMA = GAMMA
        self.ALPHA = ALPHA
        self.EPS = EPS
        self.weightReduction = weightReduction
        poleThetaSpace = np.linspace(-0.20943951, 0.20943951, 10)
        poleThetaVelSpace = np.linspace(-4, 4, 10)
        cartPosSpace = np.linspace(-2.4, 2.4, 10)
        cartVelSpace = np.linspace(-4, 4, 10)
        def getState(observation):
            cartX, cartXdot, cartTheta, cartThetadot = observation
            cartX = int(np.digitize(cartX, cartPosSpace))
            cartXdot = int(np.digitize(cartXdot, cartVelSpace))
            cartTheta = int(np.digitize(cartTheta, poleThetaSpace))
            cartThetadot = int(np.digitize(cartThetadot, poleThetaVelSpace))

            return (cartX, cartXdot, cartTheta, cartThetadot)
        
        self.getState = getState

        self.states = []
        for i in range(len(cartPosSpace)+1):
            for j in range(len(cartVelSpace)+1):
                for k in range(len(poleThetaSpace)+1):
                    for l in range(len(poleThetaVelSpace)+1):
                        self.states.append((i,j,k,l))

        self.initQ()

    def initQ(self):
        self.Q = {}
        for state in self.states:
            for action in range(2):
                self.Q[state, action] = 0

    def updateEps(self):
        if self.EPS - 2 / self.weightReduction > 0:
            self.EPS -= 2 / self.weightReduction
        else:
            self.EPS = 0

    def chooseAction(self, state):    
        values = np.array([self.Q[state,a] for a in range(2)])
        action = np.argmax(values)
        return action

    def train(self):
        observation = self.env.reset()        
        state = self.getState(observation)
        rand = np.random.random()
        action = self.chooseAction(state) if rand < (1-self.EPS) else self.env.action_space.sample()
        done = False
        rewards = 0
        while not done:
            observation_, reward, done, info = self.env.step(action)
            rewards += reward
            state_ = self.getState(observation_)
            rand = np.random.random()
            action_ = self.chooseAction(state_) if rand < (1-self.EPS) else self.env.action_space.sample()            
            self.Q[state,action] = self.Q[state,action] + self.ALPHA*(reward + self.GAMMA*self.Q[state_,action_] - self.Q[state,action])
            state, action = state_, action_            
        
        self.updateEps()
        return rewards
