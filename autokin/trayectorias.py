import torch


def coprime_sines(n_dim, n_points, densidad=0, base_frec=0):
    # https://www.desmos.com/calculator/m4pjhqjgz6
    """
    Genera trayectoria paramétrica explorando cada dimensión
    con sinusoides de frecuencias coprimas.

    El muestreo de estas curvas suele concentrarse en los
    límites del espacio, y pasa por múltiples
    coordenadas con valor 0, por lo que podría atinarle a
    las singularidades de un robot si se usan las curvas
    en el espacio de parámetros.
    """
    coefs = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
             31, 37, 41, 43, 47, 53, 59, 61, 67]

    # Acotar valores de densidad demasiado altos
    densidad = min(len(coefs)-n_dim, densidad)

    coefs = torch.tensor(coefs) * 2*torch.pi
    points = torch.zeros((n_points, n_dim))

    t = torch.linspace(0, 1, n_points)
    t += 0.5 * torch.rand((n_points)) / n_points
    #points = 0.3*torch.rand((n_dim, n_points))

    for i in range(n_dim):
        frec = base_frec + coefs[i+densidad]
        points[:, i] = torch.sin(frec*t) /2 + 0.5
    return points