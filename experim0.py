from typing import OrderedDict
import torch
from torch import nn
from torch import functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# args
n_samples = 10
batch_size = 10
epochs = 10000
learn_rate = 3e-4
n_mid_layers = 3
mid_layer_size = 10
bias = True
crack_norm = True
view_plot = True

class Regressor(nn.Module):

    def __init__(self, input_dim=1, output_dim=1,
                 n_mid_layers=1, mid_layer_size=10):
        super().__init__()
        #layers = OrderedDict()
        self.model = nn.Sequential(
            nn.Linear(input_dim, mid_layer_size, bias=bias),
            nn.Tanh(),
            nn.Linear(mid_layer_size, mid_layer_size, bias=bias),
            nn.Tanh(),
            nn.Linear(mid_layer_size, output_dim, bias=bias)
        )

    def forward(self, x):
        if crack_norm:
            out = (x+10)/20
        out = self.model(out)
        return out


model = Regressor(1, 1, 3, 10)

x = torch.linspace(-1, 1, n_samples).view(-1, 1)

def f(x): return 3*x**3
y = f(x)

train_set = TensorDataset(x, y)#(x.view(-1,1), y.view(-1,1))
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

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

if view_plot:
    x_plot = torch.linspace(-1, 1, n_samples).view(-1,1)

    with torch.no_grad():
        pred = model(x_plot)
    fig, ax = plt.subplots()
    ax.plot(x_plot, f(x))
    ax.plot(x_plot, pred)
    ax.scatter(x, y)
    ax.legend(['Target F',
               'Model',
               'Trainset'])

    plt.show()