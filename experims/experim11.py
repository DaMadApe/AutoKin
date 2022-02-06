"""
Pruebas de trayectorias para explorar el espacio de parámetros
"""
import numpy as np # TODO: Revisar si cambiar por torch

def n_hilbert(n_dim, n_points, res=1):
    """
    Devuelve una curva de Hilbert n_dimensional,
    restringida a un hipercubo unitario.
    """
    return np.zeros(n_points, n_dim)    


def coprime_sines(n_dim, n_points, wiggle=0):
    # https://www.desmos.com/calculator/m4pjhqjgz6
    """
    El muestreo de estas curvas suele concentrarse en los
    límites del espacio, y pasa por múltiples
    coordenadas con valor 0, por lo que podría atinarle a
    las singularidades de un robot si se usan las curvas
    en el espacio de parámetros.
    """
    coefs = [5, 7, 11, 13, 17, 19, 23, 29, 31]
    coefs = np.array(coefs) * 2*np.pi
    points = np.stack([np.linspace(0, 1, n_points)] * n_dim)
    for i in range(n_dim):
        points[i] = np.sin(coefs[i+wiggle]*points[i])
    return points

# Derivada de coprime_sines
def coprime_sines_dif(n_dim, n_points, wiggle=0):
    pass


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    curve = coprime_sines(2, 300)
    plt.scatter(*curve)
    plt.show()