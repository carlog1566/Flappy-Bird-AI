import torch
import random
import numpy as np
from gameAI import FlappyBird, Pipe
from model import Linear_QNet, QTrainer

LEARNING_RATE = 0.001

class Agent:
    
    # AGENT CONTRUCTOR
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.model = Linear_QNet(6, 256, 2)  # States: x dist from pipe, y dist from pipe, vertical distance to bottom of top pipe, vertical distance to top of bottom pipe, player y, player y velocity; Output: jump or don't jump
        self.trainer = QTrainer(self.model, learning_rate=LEARNING_RATE, gamma=self.gamma)


    # GET STATE FUNCTION
    def get_state(self, game):
        pass

    
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