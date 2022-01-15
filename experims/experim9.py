"""
Automatizar el entrenamiento de múltiples
robots para comparar el efecto de distintas
arquitecturas de la red neuronal.
"""
import numpy as np
import roboticstoolbox as rtb
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from experimR import RoboKinSet


class FKRegressionTask(pl.LightningModule):

    def __init__(self, model, robot, n_per_q,
                 batch_size=64, lr=1e-3, optim=torch.optim.Adam):
        super().__init__()
        self.model = model
        # Para save_graph
        self.example_input_array = torch.zeros(1, self.model.input_dim)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = self.hparams.optim(self.parameters(), lr=self.hparams.lr)
        return optimizer
        
    def training_step(self, batch, batch_idx):
        point, target = batch
        pred = self(point)
        loss = F.mse_loss(pred, target)
        self.log('train_loss', loss)
        self.log('hp_metric', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        point, target = batch
        pred = self(point)
        val_loss = F.mse_loss(pred, target)
        self.log('val_loss', val_loss)
        return val_loss

    def train_dataloader(self):
        ns_samples = [self.hparams.n_per_q] * self.hparams.robot.n
        train_set = RoboKinSet.grid_sampling(self.hparams.robot, ns_samples)
        loader = DataLoader(train_set, batch_size=self.hparams.batch_size, shuffle=True)
        return loader

    def val_dataloader(self):
        n_samples = self.hparams.n_per_q ** self.hparams.robot.n // 5
        val_set = RoboKinSet(self.hparams.robot, n_samples)
        loader = DataLoader(val_set, batch_size=self.hparams.batch_size)
        return loader


def random_robot(min_DH, max_DH, p_P):
    # rtb.DHLink([d, alpha, theta, a, joint_type])
    # rev=0, prism=1
    links = []

    for n_joint in range(np.random.randint(2, 10)):

        DH_vals = (np.random.rand(4) - min_DH) / (max_DH - min_DH)
        d, alpha, theta, a = DH_vals
        is_prism = np.random.rand() < p_P

        if is_prism:
            links.append(rtb.DHLink(alpha=alpha,theta=theta, a=a, sigma=1))
        else:
            links.append(rtb.DHLink(d=d, alpha=alpha, a=a, sigma=0))


    return rtb.DHRobot(links)




if __name__ == "__main__":

    from experim0 import MLP
    from experim13 import ResNet

    min_DH = np.array([0, 0, 0, 0] )
    max_DH = np.array([2*np.pi, 2, 2*np.pi, 2])
    prob_prism = 0.5

    robot = random_robot(min_DH, max_DH, prob_prism)

    """
    fkine_all devuelve la transformación para cada junta, por lo que
    podría hacer todos los robots de 9 juntas, y aprovechar la función
    para sacar también datos de los subconjuntos de la cadena cinemática
    """
    # q = np.random.rand(100, robot.n)
    # robot.fkine_all(q).t
    # robot.plot(q)


    input_dim = robot.n
    output_dim = 3

    mlp_params_0 = {"input_dim": input_dim, 
                    "output_dim": output_dim,
                    "depth": 3,
                    "mid_layer_size": 10,
                    "activation": torch.tanh}

    for model in [MLP(**mlp_params_0), ResNet()]:
        task = FKRegressionTask(model)

        trainer = pl.Trainer()
        trainer.fit(task)