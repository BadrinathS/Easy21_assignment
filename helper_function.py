import random

class Card():
    def __init__(self, draw_black = False):
        
        self.value = random.randint(1,10)
        if random.randint(1,3) != 2 or draw_black:
            self.is_black = True
        else:
            self.is_black = False
            self.value = -self.value
        self.is_red = not self.is_black

class State():
    def __init__(self, dealer, player):  #Takes dealer's first card and player's sum as input.
        self.dealer = dealer.value
        self.player = player.value
        self.is_terminal = False


class Environment():
    
    def start_state(self):
        s = State(Card(True), Card(True))

        return s
    
    def take_step(self, state, action):

        reward = 0

        if action == 'stick':
            while not state.is_terminal:
                state.dealer += Card().value
                if state.dealer >21 or state.player < 1:
                    state.is_terminal = True
                    reward = 1
                
                elif state.dealer > 17:
                    if state.dealer > state.player :
                        reward = -1
                    elif state.dealer < state.player:
                        reward = 1
                    state.is_terminal = True

        if action == 'hit':
            state.player += Card().value
            if state.player < 1 or state.player > 21:
                state.is_terminal = True
                reward = -1

        return state, reward
