"""
Aprender CI directamente, invirtiendo los ejemplos
usados anteriormente para entrenar modelos de CD.
"""
import torch
from torch.utils.data import DataLoader
import roboticstoolbox as rtb
from tqdm import tqdm

from experim0 import MLP
from experim1 import RoboKinSet

"""
args
"""
depth = 10
mid_layer_size = 10
activation = torch.relu
lr = 1e-3
batch_size = 512
epochs = 500

robot = rtb.models.DH.Cobra600()

input_dim = 3
output_dim = robot.n

"""
Conjunto de datos
"""
n_per_q = 10
n_samples = n_per_q ** robot.n

ns_samples = [n_per_q] * robot.n
train_set = RoboKinSet.grid_sampling(robot, ns_samples)

val_set = RoboKinSet(robot, n_samples//5)

train_loader = DataLoader(train_set,
                          batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set,
                        batch_size=batch_size, shuffle=True)

"""
Entrenamiento
"""
model = MLP(input_dim, output_dim,
                    depth, mid_layer_size,
                    activation)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

progress = tqdm(range(epochs), desc='Training')
for _ in progress:
    # Train step
    # Invertir muestras para fines de exp
    for Y, X in train_loader:
        model.train()
        pred = model(X)
        loss = criterion(pred, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Val step
    with torch.no_grad():
        for Y, X in val_loader:
            model.eval()
            pred = model(X)
            val_loss = criterion(pred, Y)

    progress.set_postfix(Loss=loss.item(), Val=val_loss.item())

# Guardar modelo
torch.save(model, 'models/experim4_v1.pt')