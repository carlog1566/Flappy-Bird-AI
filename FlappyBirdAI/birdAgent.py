# IMPORT LIBRARIES
import torch
import random
import numpy as np
from gameAI import FlappyBird, Pipe
from model import Linear_QNet, QTrainer

# DEFINE LEARNING RATE
LEARNING_RATE = 0.001


class Agent:
    
    # AGENT CONTRUCTOR
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.model = Linear_QNet(4, 256, 2)  # States: x dist from pipe, y dist to center gap, player y, player y velocity; Output: jump or don't jump
        self.trainer = QTrainer(self.model, learning_rate=LEARNING_RATE, gamma=self.gamma)


    # GET STATE FUNCTION
    def get_state(self, game):
        bird = game.player

        # Calculate and normalize y and velocity
        bird_y = bird.rect.centery / game.h
        bird_velocity = bird.y_velocity / 10

        # Store all pipe pairs ahead of the bird
        pipes_ahead = [pipe for pipe in game.pipes if pipe.rect.right > bird.rect.left]

        # If no pipes, return default state
        if not pipes_ahead:
            state = [0, 0, bird_y , bird_velocity]
            return state
        
        # Store top and bottom pipe
        next_pipe = min(pipes_ahead, key=lambda pipe: pipe.rect.left)
        for pipe in pipes_ahead:
            if (pipe.rect.left == next_pipe.rect.left) and (pipe.pos != next_pipe.pos):
                if pipe.pos == 1:
                    top_pipe = pipe
                    bottom_pipe = next_pipe
                else:
                    top_pipe = next_pipe
                    bottom_pipe = pipe

        # Calculate and normalize x and y dist to pipe/center gap
        center_gap = (top_pipe.rect.bottom + bottom_pipe.rect.top) / 2
        x_dist = (top_pipe.rect.left - bird.rect.right) / game.w
        y_dist = (center_gap - bird.rect.centery) / game.h

        # Store and return state as np array
        state = [x_dist, y_dist, bird_y, bird_velocity]

        return np.array(state, dtype=float)
         
    
    # GET ACTION FUNCTION
    def get_action(self, state):
        pass


# AI AGENT TRAINING FUNCTION
def train():
    scores = []
    record = 0
    agent = Agent()
    game = FlappyBird()

    while True:
        pass