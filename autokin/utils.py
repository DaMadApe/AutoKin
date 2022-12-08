import torch
from torch.nn.functional import pad
from torch.utils.data import random_split

from tensorboard import program

import roboticstoolbox as rtb


class RobotExecError(Exception):
    """
    Excepción levantada cuando hay un error durante la ejecución 
    de fkine para algún robot que prevenga obtener resultados.
    """
    pass


def abrir_tb(log_dir):
    tb = program.TensorBoard()
    tb.configure(logdir=log_dir, port=6006)
    tb.main()


def random_robot(min_DH: list[float] = None,
                 max_DH: list[float] = None,
                 p_P: float = 0.5,
                 min_n: int = 2,
                 max_n: int = 9,
                 n: int = None) -> rtb.DHRobot:
    """
    Robot creado a partir de parámetros DH aleatorios.
    
    args:
    min_DH (list) : Mínimos valores posibles de [d, alpha, theta, a]
    max_DH (list) : Máximos valores posibles de [d, alpha, theta, a]
    p_P (float) : Probabilidad de una junta prismática
    min_n (int) : Mínimo número posible de juntas
    max_n (int) : Máximo número posible de juntas
    n (int) : Número de juntas; si se define, se ignora min_n, max_n
    """
    # rtb.DHLink([d, alpha, theta, a, joint_type])  rev=0, prism=1

    if min_DH is None:
        min_DH = [0, 0, 0, 0]
    if max_DH is None:
        max_DH = [1, 2*torch.pi, 2*torch.pi, 1]

    min_DH = torch.tensor(min_DH)
    max_DH = torch.tensor(max_DH)

    if torch.any(min_DH > max_DH):
        raise ValueError('Parámetros mínimos de DH no son menores a los máximos')

    links = []

    if n is not None:
        n_joints = n
    else:
        n_joints = torch.randint(min_n, max_n+1, (1,))

    for _ in range(n_joints):
        DH_vals = torch.rand(4) * (max_DH - min_DH) + min_DH
        d, alpha, theta, a = DH_vals
        is_prism = torch.rand(1) < p_P

        if is_prism:
            links.append(rtb.DHLink(alpha=alpha,theta=theta, a=a, sigma=1,
                                    qlim=[0, 1.5*max_DH[0]]))
        else:
            links.append(rtb.DHLink(d=d, alpha=alpha, a=a, sigma=0))
                         #qlim=np.array([0, 1.5*max_DH[0]])))
    return rtb.DHRobot(links)


def rand_split(dataset, proportions: list[float]):
    """
    Reparte el conjunto de datos en segmentos aleatoriamente
    seleccionados, acorde a las proporciones ingresadas.

    args:
    dataset (torch Dataset): Conjunto de datos a repartir
    proportions (list[float]): Porcentaje que corresponde a cada partición
    """
    if round(sum(proportions), ndigits=2) != 1:
        raise ValueError('Proporciones ingresadas deben sumar a 1 +-0.01')
    split = [round(prop*len(dataset)) for prop in proportions]

    # HACK: Compensa por algunos valores que no suman la longitud original
    split[0] += (len(dataset) - sum(split))

    return random_split(dataset, split)


def restringir(q: torch.Tensor) -> torch.Tensor:
    """
    Para una lista de actuación q [mxn], limita cada uno de los m puntos a la
    n-esfera unitaria, y evitar que dos actuadores tensen al máx. simultánemante
    """
    def _map(u, v, s=0, t=0): 
        return u*torch.sqrt(1 - v**2/2 - s**2/2 - t**2/2 
                              + v**2*s**2/3 + s**2*t**2/3 + t**2*v**2/3
                              - v**2*s**2*t**2/4)
    trans_q = torch.zeros_like(q)
    n = q.shape[-1]
    for i in range(n):
        cols = (q[:,(j+i)%n] for j in range(n))
        trans_q[:,i] = _map(*cols)
    return trans_q


def linterp(q1: torch.Tensor, q2: torch.Tensor, pasos: int) -> torch.Tensor:
    """
    Para dos puntos de actuación q1, q2 de forma [1xn], producir una
    interpolación lineal entre q1 y q2, con forma [pasos x n].
    """
    assert q1.size() == q2.size(), 'Tensores de distinto tamaño'
    assert len(q1.size()) == 1, 'Tensores de más de 1D'

    n_dim = q1.size()[-1]
    interp = torch.zeros((pasos, n_dim))

    for i in range(n_dim):
        qi, qf = q1[i], q2[i]
        interp[:, i] = torch.linspace(qi, qf, pasos)

    return interp


def suavizar(q: torch.Tensor,
             q_prev: torch.Tensor,
             dq_max: float) -> torch.Tensor:
    """
    Para lista de actuación q de forma [mxn], aplica linterp entre puntos consecutivos
    para evitar que el cambio instantáneo de algún actuador exceda dq_max.
    """
    if len(q_prev.shape) == 1:
        q_prev = q_prev.unsqueeze(0)

    # q = torch.cat([q_prev, q])
    q = torch.cat([q_prev, q])
    # Tomar la norma de la segunda diferencia de q
    q_diff = q.diff(n=2, dim=0).norm(dim=-1)
    # Tomar los puntos en los que la norma de la diferencia excede dq_max
    oversteps = (q_diff/dq_max).round().int()
    # Agregar 0 al inicio para ajustar dimensiones
    oversteps = pad(oversteps, (1,0), 'constant', 0)
    # Arreglos auxiliares para seguir índices luego de insertar interpolaciones
    total_extra_steps = oversteps.sum().item()
    accum_steps = oversteps.cumsum(dim=0)
    accum_steps = torch.cat([torch.zeros(1,dtype=int), accum_steps])

    q_ext = torch.zeros(q.shape[0]+total_extra_steps, q.shape[1])

    for i in range(len(q)-1):
        cur_idx = i + accum_steps[i].item()

        if oversteps[i] == 0:
            q_ext[cur_idx] = q[i]
        else:
            interp_q = linterp(q[i], q[i+1], oversteps[i]+2)
            q_ext[cur_idx:cur_idx+oversteps[i]+2] = interp_q

    q_ext[-1] = q[-1]

    return q_ext


if __name__ == "__main__":
    a = torch.rand(3,2)
    da = a.diff(dim=0)
    max_da = 0.7

    q_diff = a.diff(n=2, dim=0).norm(dim=-1)
    oversteps = (q_diff/max_da).round().int()
    oversteps = torch.cat([#oversteps[0].unsqueeze(0), 
                           oversteps,
                           oversteps[-1].unsqueeze(0)])
    print(a)
    print(oversteps)
    print(suavizar(a, torch.zeros(2), max_da))