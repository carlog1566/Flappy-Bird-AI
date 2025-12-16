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
        pass