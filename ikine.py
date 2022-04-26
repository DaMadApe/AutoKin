from typing import Callable

import torch
from torch.autograd.functional import jacobian


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
def ikine_pi_jacob(q_start: torch.Tensor, p_target: torch.Tensor,
                   fkine: Callable, jacob : Callable,
                   eta=0.01, max_error=0, max_iters=1000):
    """
    Método de pseudoinverso de jacobiano
    q_start (Tensor) : Configuración inicial del robot
    p_target (Tensor) : Posición objetivo en espacio cartesiano
    fkine (Callable) : Función de cinemática directa (f: q -> p)
    jacob (Callable) : Función que devuelve jacobiano (f: q -> dp/dq)
    eta (float) : Coeficiente de actualización por iteración
    max_error (float) : Máxima distancia tolerada entre posición y
        objetivo, sujeto a max_iters
    max_iters (int) : Máximo número de iteraciones
    """
    q = q_start.clone().detach()
    for _ in range(max_iters):
        delta_x = p_target - fkine(q)
        error = torch.linalg.norm(delta_x)
        if error < max_error:
            break

        pi_jacob = torch.linalg.pinv(jacob(q))
        q_update = eta * torch.matmul(pi_jacob, delta_x)

        q += q_update

    return q


def ikine_trans_jacob(q_start: torch.Tensor, p_target: torch.Tensor,
                      fkine: Callable, jacob : Callable,
                      eta=0.01, max_error=0, max_iters=1000):
    """
    Método de transpuesta de jacobiano
    q_start (Tensor) : Configuración inicial del robot
    p_target (Tensor) : Posición objetivo en espacio cartesiano
    fkine (Callable) : Función de cinemática directa (f: q -> p)
    jacob (Callable) : Función que devuelve jacobiano (f: q -> dp/dq)
    eta (float) : Coeficiente de actualización por iteración
    max_error (float) : Máxima distancia tolerada entre posición y
        objetivo, sujeto a max_iters
    max_iters (int) : Máximo número de iteraciones
    """
    q = q_start.clone().detach()
    for _ in range(max_iters):
        delta_x = p_target - fkine(q)
        error = torch.linalg.norm(delta_x)
        if error < max_error:
            break

        trans_jacob = jacob(q).T

        q_update = eta * torch.matmul(trans_jacob, delta_x)
        q += q_update

    return q


"""
import nevergrad as ng
def ikine_ngopt(q_start: torch.Tensor, p_target: torch.Tensor,
                fkine: Callable, # jacob : Callable,
                eta=0.01, max_error=0, max_iters=1000):

    def error(q):
        return torch.sum((fkine(q) - p_target)**2)

    optim = ng.optimizers.NGOpt(parametrization=len(q_start), budget=100)
    q = optim.minimize(error)
"""



def ikine_adam(q_start: torch.Tensor, p_target: torch.Tensor,
               fkine: Callable, # jacob : Callable,
               max_error=0, max_iters=1000, **adam_kwargs):

    current_p = fkine(q_start)
    current_q = q_start.detach().clone()
    current_q.requires_grad = True

    def error(p):
        return torch.sum((p - p_target)**2)

    optim = torch.optim.Adam([current_q], **adam_kwargs)

    for _ in range(max_iters):
        current_p = fkine(current_q)
        current_error = error(current_p)

        if current_error < max_error:
            break

        optim.zero_grad()
        current_error.backward()
        optim.step()

    return current_q.detach()