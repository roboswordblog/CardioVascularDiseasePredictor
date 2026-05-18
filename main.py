import pandas as pd
import torch.nn as nn
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


df = pd.read_csv('cardio_data.csv', sep=';')
df["age_years"] = df["age"] / 365.25
df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)
df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]
df["map"] = df["ap_lo"] + (df["pulse_pressure"] / 3)

df["high_bp"] = ((df["ap_hi"] >= 140) | (df["ap_lo"] >= 90)).astype(int)
df["chol_gluc_risk"] = df["cholesterol"] + df["gluc"]
df = df[(df["ap_hi"] >= 80) & (df["ap_hi"] <= 250)]
df = df[(df["ap_lo"] >= 40) & (df["ap_lo"] <= 200)]
df = df[df["ap_hi"] > df["ap_lo"]]
X = df.drop(labels=["id", "cardio"], axis=1)
y = df["cardio"]
X = torch.FloatTensor(X.values)
Y = torch.FloatTensor(y.values)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(17, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 8)
        self.out = nn.Linear(8, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.out(x)
        return x


X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, shuffle=True, random_state=42
)

mean = X_train.mean(dim=0)
std = X_train.std(dim=0)

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std
y_train = y_train.long()
y_test = y_test.long()
model = Model()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00125)
train_ds = TensorDataset(X_train, y_train)
loader = DataLoader(train_ds, batch_size=512, shuffle=True)
for epoch in range(45000):
    for xb, yb in loader:
        pred = model(xb)
        loss = criterion(pred, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

with torch.no_grad():
    model.eval()
    test_outputs = model(X_test)
    predictions = torch.argmax(test_outputs, dim=1)
    accuracy = (predictions == y_test).float().mean()
    print(f"Test Accuracy: {accuracy.item():.4f}")
with torch.no_grad():
    model.eval()