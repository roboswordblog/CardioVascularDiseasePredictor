import pandas as pd
import torch.nn as nn
import torch
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('cardio_data.csv', sep=';')

X = dataset.drop(labels=["id", "cardio"], axis=1)
y = dataset["cardio"]
X = torch.FloatTensor(X.values)
Y = torch.FloatTensor(y.values)


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


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)

y_train = y_train.long()
y_test = y_test.long()
model = Model()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.004)
epochs = 50000

for i in range(epochs):
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 10 == 0:
        predictions = torch.argmax(y_pred, dim=1)
        accuracy = (predictions == y_train).float().mean()
        print(accuracy)
