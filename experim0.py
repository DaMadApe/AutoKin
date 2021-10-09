from typing import OrderedDict
import torch
from torch import nn
from torch import functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# args
n_samples = 50
batch_size = 5
epochs = 10
learn_rate = 3e4
n_mid_layers = 3
mid_layer_size = 256

class Regressor(nn.Module):

    def __init__(self, input_dim=1, output_dim=1,
                 n_mid_layers=1, mid_layer_size=10):
        super().__init__()
        #layers = OrderedDict()
        self.model = nn.Sequential(
            #nn.Flatten(start_dim=1),
            nn.Linear(1, mid_layer_size),
            nn.ReLU(),
            nn.Linear(mid_layer_size, mid_layer_size),
            nn.ReLU(),
            nn.Linear(mid_layer_size, 1),
            nn.Softmax()
        )

    def forward(self, x):
        out = (x+10)/20
        #out = out.view(-)
        out = self.model(out)
        return out


model = Regressor(1, 1, 3, 10)

x = torch.linspace(-10, 10, n_samples)

def f(x): return 3*x**3
y = f(x)

train_set = TensorDataset(x, y)#(x.view(-1,1), y.view(-1,1))
train_loader = DataLoader(train_set, batch_size=batch_size)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
progress = tqdm(range(epochs), desc='Training')

for _ in progress:
    for X, Y in train_loader:
        pred = model(X)
        loss = criterion(pred, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        progress.set_postfix(Loss=loss.item())

with torch.no_grad():
    pred = model(x)
fig, ax = plt.subplots()
ax.scatter(x, y)
ax.plot(x, pred)

plt.show()