import pandas as pd
import torch.nn as nn
import torch

dataset = pd.read_csv('cardio_data.csv', sep=';')

X = dataset.drop(labels=["id", "cardio"])
y = dataset["cardio"]


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(11, 16)
        self.fc2 = nn.Linear(16, 8)
        self.out = nn.Linear(8, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.out(x)
        return x
