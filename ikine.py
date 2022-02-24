import torch
from torch.autograd.functional import jacobian
import roboticstoolbox as rtb

#TODO: Cambiar todo numpy por torch para acelerar interacción con redes
import numpy as np


def batch_jacobian(func, batch, create_graph=False, vectorize=False):
    """
    Función para calcular el jacobiano para cada muestra de un batch
    https://discuss.pytorch.org/t/computing-batch-jacobian-efficiently/80771/4

    args:
    func (Callable) : Función a la que se le saca Jacobiano
    batch (nd-array) : Conjunto de puntos con forma (N_muestras, dim_muestra)
    """
    def _func_sum(batch):
        return func(batch).sum(dim=0)
    return jacobian(_func_sum, batch,
                    create_graph=create_graph,
                    vectorize=vectorize).permute(1,0,2)

"""
Métodos iterativos de cinemática inversa
"""
def ikine_pi_jacob(q_start, x_target, fkine, jacob,
                   eta=0.01, min_error=0, max_iters=1000):
    """
    Método de pseudoinverso de jacobiano

    """
    q = np.copy(q_start)
    for _ in range(max_iters):
        delta_x = x_target - fkine(q)
        error = np.linalg.norm(delta_x)
        if error < min_error:
            break

        pi_jacob = np.linalg.pinv(jacob(q))
        q_update = eta * np.dot(pi_jacob, delta_x)

        q += q_update

    return q


# Método de transpuesta de jacobiano
def ikine_trans_jacob(q_start, x_target, fkine, jacob,
                      eta=0.01, min_error=0, max_iters=1000):
    q = np.copy(q_start)
    for _ in range(max_iters):
        delta_x = x_target - fkine(q)
        error = np.linalg.norm(delta_x)
        if error < min_error:
            break

        trans_jacob = jacob(q).T

        q_update = eta * np.dot(trans_jacob, delta_x)
        q += q_update

    return q


if __name__ == "__main__":
    
    from utils import denorm_q

    robot = rtb.models.DH.Cobra600()

    q0 = denorm_q(robot, [0.1, 0.1, 0.1, 0.1])

    q_target = denorm_q(robot, [0.1, 0.4, 0.2, 0.3])
    x_target = robot.fkine(q_target).t

    q_inv = ikine_pi_jacob(q0, x_target,
                           lambda q: robot.fkine(q).t, 
                           lambda q: robot.jacob0(q)[:3],
                           max_iters=500)

    print(f"""
        Initial q: {q0}
        Initial x: {robot.fkine(q0).t}
        Requested q: {q_target}
        Requested x: {x_target}
        Found q: {q_inv}
        Reached x: {robot.fkine(q_inv).t}""")