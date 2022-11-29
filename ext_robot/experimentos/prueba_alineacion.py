import os
import torch
import numpy as np
from matplotlib import pyplot as plt

# ext_ds_dir = os.path.join('gui', 'app_data', 'robots', 'r3c', 'datasets')
sofa_ds_dir = os.path.join('gui', 'app_data', 'robots', 'LSL', 'datasets')

def load_last_ds(dir):
    return torch.load(os.path.join(dir, sorted(os.listdir(dir))[-1]))

d_set = load_last_ds(sofa_ds_dir)

q_set = np.concatenate([d_point[0].unsqueeze(0).numpy() for d_point in d_set])
p_set = np.concatenate([d_point[1].unsqueeze(0).numpy() for d_point in d_set])

q_diff = np.linalg.norm(np.diff(q_set, axis=0), axis=-1)
p_diff = np.linalg.norm(np.diff(p_set, axis=0), axis=-1)

t = np.arange(len(q_diff))

fig = plt.figure()
ax = fig.add_subplot()

# ax.scatter(t, q_diff, color='orangered')
ax.plot(t, q_diff, color='orange')

# ax.scatter(t, p_diff, color='royalblue')
ax.plot(t, 0.1*p_diff, color='lightblue')

plt.tight_layout()
plt.show()