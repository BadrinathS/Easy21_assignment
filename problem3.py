import random
import numpy as np
from helper_function import *
from problem2 import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm


'''
Reference from 

https://www.geeksforgeeks.org/sarsa-reinforcement-learning/

https://github.com/boranzhao/Easy21
'''



dealer_count = 10
player_count = 21
action_count = 2

def get_action_int(action):
    if action == 'hit':
        return 0
    else:
        return 1

class Sarsa():
    def __init__(self, environment, N0, lam):
        self.env = environment
        self.lam = lam
        self.N0 = N0
        self.gamma = 1
        self.mse = float('inf')
        self.N = np.zeros((dealer_count, player_count, action_count))

        self.Q = np.zeros((dealer_count, player_count, action_count))
        self.V = np.zeros((dealer_count, player_count))

        self.count_win = 0      #Using to calculate win percentage
        self.iteration = 0      #Using to calculate win percentage

    def return_action(self, state):
        
        num_visits = sum(self.N[state.dealer-1, state.player-1, :])
        eps = self.N0/(self.N0 + num_visits)

        if random.random() < eps:
            if random.random() < 0.5:
                return 'hit'
            else:
                return 'stick'
        
        else:
            if np.argmax(self.Q[state.dealer-1, state.player-1,:]) == 0:
                return 'hit'
            else:
                return 'stick'

    
    def train_game(self, iteration):

        for episode in range(iteration):

            state = self.env.start_state()
            action = get_action_int(self.return_action(state))
            
            while not state.is_terminal:
                self.N[state.dealer-1, state.player-1,action] += 1

                state_new, reward = self.env.take_step(state, action)
                if state_new.is_terminal:
                    Q_new = 0
                else:
                    action_new = get_action_int(self.return_action(state_new))
                    Q_new = self.Q[state_new.dealer-1, state_new.player-1, action_new]


                alpha = 1/self.N[state.dealer-1, state.player-1, action]

                error = reward + self.gamma*Q_new - self.Q[state.dealer-1, state.player-1, action]
                self.Q += alpha*error

                state.dealer = state_new.dealer
                state.player = state_new.player
                state.is_terminal = state_new.is_terminal

                if not state.is_terminal:
                    action = action_new
            
            if reward == 1:
                self.count_win += 1
            
        self.iteration += iteration
    
    def update_value_function(self):
        for dealer in range(self.V.shape[0]):
            for player in range(self.V.shape[1]):
                self.V[dealer, player] = max(self.Q[dealer, player, :])
    
N0 = 100

mc_agent = Monte_Carlo(Environment(), N0)
mc_agent.train_game(1000000)

print('After %s episodes, winning percentage: %f' %(mc_agent.iteration, mc_agent.count_win/mc_agent.iteration))
Qmc = mc_agent.Q


Lambda = np.linspace(0,1,10)

mse = []
for i in range(len(Lambda)):
    mse.append([])

for lam_id, lam in enumerate(Lambda):
    agent = Sarsa(Environment(), N0, lam)
    for i in range(1000):
        print(i)
        agent.train_game(1)
        agent.mse = np.mean((agent.Q-Qmc)**2)
        mse[lam_id].append(agent.mse)

        print('lambda = %s, MSE: %f, winning percentage: %f' %(agent.lam, agent.mse, agent.count_win/agent.iteration))
    

X = list(range(1,len(mse[0])+1))    
fig = plt.figure('MSE against lambda')
plt.plot(Lambda, [x[-1] for x in mse])
plt.xlabel('lambda')
plt.ylabel('mean-squared error')
plt.savefig('MSE against lambda')
plt.show()


fig = plt.figure('Learning process')
plt.subplot(211)
plt.plot(X,mse[0],color = Color[0], linestyle=LineStyle[0%4])
plt.xlabel('episode')
plt.ylabel('MSE')
plt.title('lambda = 0')

plt.subplot(212)
plt.plot(X,mse[-1],color = Color[0], linestyle=LineStyle[0%4])
plt.xlabel('episode')
plt.ylabel('MSE')
plt.title('lambda = 1')

plt.savefig('Learning process for lambda 0 and 1')
plt.show()