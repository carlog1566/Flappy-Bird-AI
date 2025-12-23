# IMPORT LIBRARIES
import torch
import random
import numpy as np
from collections import deque
from gameAI import FlappyBird, Pipe
from model import Linear_QNet, QTrainer

# DEFINE CONSTANTS
MAX_MEMORY = 100000
BATCH_SIZE = 32
LEARNING_RATE = 0.0001


class Agent:
    
    # AGENT CONTRUCTOR
    def __init__(self):
        self.n_games = 0
        self.frame_count = 0
        self.epsilon_start = 1.0
        self.epsilon_end = 0.00001
        self.epsilon_decay = 500
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(4, 128, 2)  # States: x dist from pipe, y dist to center gap, player y, player y velocity; Output: don't jump or jump

        # Create a target netowrk, copy main model weights into it, set to eval mode
        self.target_model = Linear_QNet(4, 128, 2)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.trainer = QTrainer(self.model, self.target_model, learning_rate=LEARNING_RATE, gamma=self.gamma)


    # GET STATE FUNCTION
    def get_state(self, game):
        bird = game.player

        # Calculate and normalize y and velocity ([-1, 1])
        bird_velocity = (2 * (bird.y_velocity + 10) / 25) - 1    # Velocity -> [-10, 15]
        bird_y = (2 * (bird.rect.centery + 100) / 900) - 1   # y -> [-100, 800]

        # Store all pipe pairs ahead of the bird
        pipes_ahead = [pipe for pipe in game.pipes if pipe.rect.left > bird.rect.right]

        # If no pipes, return default state
        if not pipes_ahead:
            state = [0, 0, bird_y, bird_velocity]
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

        # Calculate and normalize x and y dist to pipe/center gap (x: [0, 1]; y: [-1, 1])
        x_dist = (top_pipe.rect.left - bird.rect.right) / game.w
        center_gap = (top_pipe.rect.bottom + bottom_pipe.rect.top) / 2
        y_dist = (bird_y - center_gap) / game.h

        # Store and return state as np array
        state = [x_dist, y_dist, bird_y, bird_velocity]

        return np.array(state, dtype=float)
    

    # REMEMBER FUNCTION
    def remember(self, state, action, reward, next_state, game_over):
        # Store past experience for replay training
        self.memory.append((state, action, reward, next_state, game_over))

    
    # TRAIN LONG MEMORY FUNCTION
    def train_long_memory(self):
        # Sample a random batch from replay memory; use it all if memory length is small
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        
        # Unpack the sample into separate arrays for training
        states, actions, rewards, next_states, game_overs = zip(*mini_sample)

        # Train the model on the sampled experiences
        self.trainer.train_step(states, actions, rewards, next_states, game_overs)

    
    # TRAIN SHORT MEMORY FUNCTION
    def train_short_memory(self, state, action, reward, next_state, game_over):
        # Train immediately on the most recent experience
        self.trainer.train_step(state, action, reward, next_state, game_over)


    # UPDATE TARGET NETOWRK FUNCTION
    def update_target_network(self):
        # Update target netowrk by copying latest weights from main model
        self.target_model.load_state_dict(self.model.state_dict())

    
    # GET ACTION FUNCTION
    def get_action(self, state):
        # Decay exploration rate as more games are played
        self.epsilon = max(self.epsilon_end, self.epsilon_start - (self.n_games / self.epsilon_decay))

        # Choose a random action or one based on model prediction and return it
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            return torch.argmax(prediction).item()



# AI AGENT TRAINING FUNCTION
def train():
    record = 0
    agent = Agent()
    game = FlappyBird()
    
    while True:
        # Get current game state
        state_old = agent.get_state(game)

        # Select action using epslilon-greedy policy
        final_move = agent.get_action(state_old)

        # Perform action and observe reward and game outcome
        reward, game_over, score = game.play(final_move)

        # Set state_new to a terminal state if game_over
        if game_over:
            state_new = np.zeros(4, dtype=float)
        else:
            state_new = agent.get_state(game)

        print('Move:', final_move, 'Reward:', reward)

        # Train on the most recent experience (every 5 frames)
        agent.frame_count += 1
        if agent.frame_count % 5 == 0:
            agent.train_short_memory(state_old, final_move, reward, state_new, game_over)

        # Store experience for replay training
        agent.remember(state_old, final_move, reward, state_new, game_over)

        if game_over:
            # Reset game and update episode count
            game.reset()
            agent.n_games += 1

            # Train on a batch of past experiences
            agent.train_long_memory()

            # Update target network every 25 games
            if agent.n_games % 25 == 0:
                agent.update_target_network()
            
            # Save model if a new high score is achieved
            if score > record:
                record = score
                agent.model.save()

            print('Game:', agent.n_games, 'Score:', score, 'Record:', record)


if __name__ == '__main__':
    train()
