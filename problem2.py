'''
Reference 
https://github.com/analog-rl/Easy21
'''


import numpy as np
import random
from helper_function import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm



dealer_count = 10
player_count = 21
action_count = 2
# Hit is defined as 0 and stick is defined as 1

def get_action_int(action):
    if action == 'hit':
        return 0
    else:
        return 1

class Monte_Carlo():
    def __init__(self, environment, N0):
        self.env = environment

        self.N = np.zeros((dealer_count, player_count, action_count))

        self.Q = np.zeros((dealer_count, player_count, action_count))
        self.V = np.zeros((dealer_count, player_count))

        self.N0 = N0
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
            sequence_state_action = []

            state = self.env.start_state()

            while not state.is_terminal:

                action = self.return_action(state)
                sequence_state_action.append((state.dealer, state.player, action))
                self.N[state.dealer-1, state.player-1, get_action_int(action)] += 1

                state, reward = self.env.take_step(state, action)
            if reward == 1:
                self.count_win += 1
            for pair in sequence_state_action:
                dealer = pair[0]
                player = pair[1]
                a = pair[2]
                step = 1/self.N[dealer-1, player-1, get_action_int(a)]
                error = reward - self.Q[dealer-1, player-1, get_action_int(a)]
                self.Q[dealer-1, player-1, get_action_int(a)] += step*error

        self.iteration += iteration
        print('Win percentage: %f'% (float(self.count_win/self.iteration)) )

        for d in range(dealer_count):
            for p in range(player_count):
                self.V[d,p] = max(self.Q[d, p,:])

            
    def plot_frame(self, ax):
        X = np.arange(0, dealer_count, 1)
        Y = np.arange(0, player_count, 1)
        X , Y = np.meshgrid(X, Y)
        Z = self.V[X,Y]

        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        return surf
    
# N0 = 100
# agent = Monte_Carlo(Environment(), N0)
# for i in range(10):
#     agent.train_game(5000)


def animate(frame):
    i = agent.iteration
    step_size = i
    step_size = max(1, step_size)
    step_size = min(step_size, 2**16)
    agent.train_game(step_size)

    ax.clear()
    surf = agent.plot_frame(ax)
    plt.title('Monte Carlo Score: %s frame: %s step_size %s ' %(float(agent.count_win/agent.iteration*100), frame, step_size))

    fig.canvas.draw()

    print("Done", frame, step_size, i)
    return surf

def play():

    N0 = 100
    agent = Monte_Carlo(Environment(), N0)
    fig = plt.figure("N100")
    ax = fig.add_subplot(111, projection='3d')

    ani = animation.FuncAnimation(fig, animate, 32, repeat=False)
    ani.save('Monte Carlo 100.gif', writer='imagepick', fps=3)



    N0 = 1
    agent = Monte_Carlo(Environment(), N0)
    fig = plt.figure("N1000")
    ax = fig.add_subplot(111, projection='3d')

    ani = animation.FuncAnimation(fig, animate, 32, repeat=False)
    ani.save('Monte Carlo 1000.gif', writer='imagepick', fps=3)

    N0 = 1
    agent = Monte_Carlo(Environment(), N0)
    fig = plt.figure("N1")
    ax = fig.add_subplot(111, projection='3d')

    ani = animation.FuncAnimation(fig, animate, 32, repeat=False)
    ani.save('Monte Carlo 1.gif', writer='imagepick', fps=3)