"""
Obtención numérica de un Jacobiano.
- Usar trayectorias de experim11
- Diferenciación manual de los puntos adyacentes

Prox Exp: Primeras pruebas de compensación de error dinámico
"""

# from experim11 import 


if __name__ == "__main__":
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