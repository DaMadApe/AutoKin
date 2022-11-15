from math import floor, ceil

import torch

import roboticstoolbox as rtb


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


def restringir(q: torch.Tensor) -> torch.Tensor:
    """
    Para una lista de actuación q [mxn], limita cada uno de los m puntos a la
    n-esfera unitaria, y evitar que dos actuadores tensen al máx. simultánemante
    """
    def _map(u, v, s, t): return u*torch.sqrt(1 - v**2/2 - s**2/2 - t**2/2 
                                                + v**2*s**2/3 + s**2*t**2/3 + t**2*v**2/3
                                                - v**2*s**2*t**2/4)

    maps = [lambda u, v: (_map(u, v, 0, 0), _map(v, u, 0, 0)),
            lambda u, v, s: (_map(u, v, s, 0), _map(v, s, u, 0), _map(s, u, v, 0)),
            lambda u, v, s, t: (_map(u, v, s, t), _map(v, s, t, u), _map(s, t, u, v), _map(t, u, v, s))]

    map_fn = maps[q.size()[-1] - 2]

    trans_q = q.clone()
    for i, point in enumerate(q):
        trans_q[i,:] = torch.tensor(map_fn(*point[:]))

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
    q = torch.cat([q_prev.unsqueeze(0), q])
    q_diff = q.diff(dim=0)

    oversteps = (q_diff.abs()/dq_max).round().int().max(dim=1).values
    total_extra_steps = oversteps.sum().item()

    accum_steps = oversteps.cumsum(dim=0)
    accum_steps = torch.cat([torch.zeros(1,dtype=int), accum_steps])

    q_ext = torch.zeros(q.shape[0]+total_extra_steps, q.shape[1])

    for i in range(len(q)-1):
        cur_idx = i + accum_steps[i].item()
        if oversteps[i] == 0:
            q_ext[cur_idx: cur_idx+2] = q[i:i+2]
        else:
            interp_q = linterp(q[i], q[i+1], oversteps[i]+2)
            q_ext[cur_idx: cur_idx+oversteps[i]+2] = interp_q

    return q_ext


def alinear_datos(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """
    Tomar dos vectores q [mq x n] y p [mp x 3], con mp >= mq, y producir de salida
    un par q_out, p_out con mismo número (mq) de puntos alineados temporalmente.

    NOTE: Se asume que no hay padding
    """
    prop = (len(p)-1)/(len(q)-1)

    p_out = torch.zeros(q.shape[0], p.shape[1])

    for i, _ in enumerate(q):
        low = floor(prop*i)
        top = ceil(prop*i)
        p_out[i] = torch.lerp(p[low], p[top], weight=prop*i%1)

    return q, p_out


if __name__ == "__main__":
    a = torch.rand(3,2)
    da = a.diff(dim=0)
    max_da = 0.7
    oversteps = (da.abs()/max_da).round().int().max(dim=1).values
    cumsteps = oversteps.cumsum(dim=0)

    print('res', suavizar(a, torch.zeros(2),max_da))