import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.optim as optim

# Read training data
train = pd.read_csv('titanic/train.csv')

# Create CNN
class CNN(torch.nn.Module):
    '''
    Constructs the CNN
    '''
    def __init__(self):
        super(CNN, self)._init__()

        self.conv1 = nn.Conv2d(train.shape[1] - 2) # -2 is to account for the extra columns
        self.fc1 = nn.Linear(train.shape[1] - 2, 1)
    
    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = self.fc1(x)

        return x
    
model = CNN() # Create the model

criterion = nn.CrossEntropyLoss() # Define Loss Function
optimizer = optim.SGD(model.parameters(), lr=0.01) # Define Optimizer