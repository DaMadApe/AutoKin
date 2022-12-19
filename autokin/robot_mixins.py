import logging

import torch
from torch.autograd.functional import jacobian

import numpy as np
# import nevergrad as ng
from scipy.optimize import differential_evolution


logger = logging.getLogger('autokin')


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


class IkineMixin:

    def __init__(self):
        super().__init__()
    
    def ikine_pi_jacob(self,
                       q_start: torch.Tensor,
                       p_target: torch.Tensor,
                       eta=0.01,
                       max_error=0,
                       max_iters=1000) -> torch.Tensor:
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
            p_current = self.fkine(q)[1]
            delta_x = p_target - p_current
            logger.debug(f"target={p_target}, current={p_current}")
            error = torch.linalg.norm(delta_x)
            if error < max_error:
                break

            pi_jacob = torch.linalg.pinv(self.jacob(q))
            q_update = eta * torch.matmul(pi_jacob, delta_x)
            logger.debug(f"q_update={q_update}")
            q += q_update
            # Restringir valores de q al intervalo [0,1]
            q = q.clamp(min=0, max=1)

        return q.detach()


    def ikine_trans_jacob(self,
                          q_start: torch.Tensor,
                          p_target: torch.Tensor,
                          eta=0.01,
                          max_error=0,
                          max_iters=1000) -> torch.Tensor:
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
            delta_x = p_target - self.fkine(q)[1]
            error = torch.linalg.norm(delta_x)
            if error < max_error:
                break

            trans_jacob = self.jacob(q).T

            q_update = eta * torch.matmul(trans_jacob, delta_x)
            q += q_update

        return q

    def ikine_de(self, 
                 q_start: torch.Tensor,
                 p_target: torch.Tensor,
                 strategy: str,
                 max_error=0,
                 max_iters=50) -> torch.Tensor:

        def error(q: np.array):
            _, p_reached = self.fkine(torch.tensor(q, dtype=torch.float32))
            logger.debug(f'p_reached = {p_reached}')
            error = torch.norm(p_reached-p_target).item()
            logger.debug(f'error = {error}')
            return error
        result = differential_evolution(error,
                                        bounds=[(0,1)]*self.n,
                                        strategy=strategy,
                                        maxiter=max_iters,
                                        atol=max_error,
                                        updating='immediate',
                                        disp=logger.level==logging.DEBUG,
                                        )
        return torch.tensor(result.x, dtype=torch.float32)

    # def ikine_ngopt(self, 
    #                 q_start: torch.Tensor,
    #                 p_target: torch.Tensor,
    #                 eta=0.01, max_error=0, max_iters=1000):

    #     def error(q: torch.Tensor):
    #         p_reached = self.fkine(torch.tensor(q, dtype=torch.float32))[1]
            
    #         return torch.norm(p_reached-p_target).item()

    #     optim = ng.optimizers.NGOpt(parametrization=len(q_start), budget=100)
    #     optim.parametrization.register_cheap_constraint(lambda x: torch.all(x<=1))
    #     q = optim.minimize(error).value
    #     return torch.tensor(q, dtype=torch.float32)

    # def ikine_adam(q_start: torch.Tensor, p_target: torch.Tensor,
    #             fkine: Callable, # jacob : Callable,
    #             max_error=0, max_iters=1000, **adam_kwargs):

    #     current_p = fkine(q_start)
    #     current_q = q_start.detach().clone()
    #     current_q.requires_grad = True

    #     def error(p):
    #         return torch.sum((p - p_target)**2)

    #     optim = torch.optim.Adam([current_q], **adam_kwargs)

    #     for _ in range(max_iters):
    #         current_p = fkine(current_q)
    #         current_error = error(current_p)

    #         if current_error < max_error:
    #             break

    #         optim.zero_grad()
    #         current_error.backward()
    #         optim.step()

    #     return current_q.detach()