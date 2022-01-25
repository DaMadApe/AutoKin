"""
Pruebas de trayectorias para explorar el espacio de parámetros
"""
import numpy as np

def n_hilbert(n_dim, n_points, res=1):
    """
    Devuelve una curva de Hilbert n_dimensional,
    restringida a un hipercubo unitario.
    """
    return np.zeros(n_points, n_dim)    


def coprime_sines(n_dim, n_points):
    #https://www.desmos.com/calculator/m4pjhqjgz6
    return np.zeros(n_points, n_dim)


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    """
    import roboticstoolbox as rtb
    
    robot = rtb.models.DH.Cobra600()

    n_points = 5

    qs = np.zeros(n_points, robot.n)
    ps = robot.fkine(qs)
    j_samples = []

    j_real = [robot.jacob0(q)[] for q in qs]

    for i in range(n_points - 1):
        dq = qs[i] - qs[i+1]
        dp = ps[i] - ps[i+1]
        j_samples.append(dp/dq) """

    curve = coprime_sines(2, 100)
    plt.plot(curve)
    plt.show()