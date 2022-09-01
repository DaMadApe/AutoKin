import torch

import roboticstoolbox as rtb


def random_robot(min_DH=None, max_DH=None, p_P=0.5, min_n=2, max_n=9, n=None):
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


def restringir(q):
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


def linterp(q1, q2, pasos):
    assert q1.size() == q2.size(), 'Tensores de distinto tamaño'
    assert len(q1.size()) == 1, 'Tensores de más de 1D'

    n_dim = q1.size()[-1]
    interp = torch.zeros((pasos, n_dim))

    for i in range(n_dim):
        qi, qf = q1[i], q2[i]
        interp[:, i] = torch.linspace(qi, qf, pasos)

    return interp