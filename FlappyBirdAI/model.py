# IMPORT LIBRARIES
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class Linear_QNet(nn.Module):

    # NETWORK CONSTRUCTOR
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize base and layers
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    # FORWARD FUNCTION
    def forward(self, x):
        # Pass input data into the neural network layers and return output
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    # SAVE FUNCTION
    def save(self, file_name='model.pth'):
        # Save final model
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)



class QTrainer:

    # INITIALIZE TRAINER
    def __init__(self, model, learning_rate, gamma):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    # TRAIN STEP FUNCTION
    def train_step(self, state, action, reward, next_state, done):
        # Convert all inputs to tensors (so that the NN can work with it)
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        
        # Convert each tensor from a 1D sample to a batch size of 1 (so that it can be processed by the NN)
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # Predict Q-values, copy for loss calculation
        pred = self.model(state)
        target = pred.clone()

        # Update Q-values using Bellman Equation
        for i in range(len(done)):
            Q_new = reward[i]
            if not done[i]:
                Q_new = reward[i] + self.gamma * torch.max(self.model(next_state[i]))

            target[i][torch.argmax(action).item()] = Q_new

        # Reset gradients, compute loss via mean squared error, backpropagate loss, update weights
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

