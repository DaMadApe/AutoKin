"""
Algoritmo de CI iterativo a partir de CD, J_q(CD)
Usar CD, J de robots DH con Corke, después probar
con una aproximación neuronal guardada y su derivada.

Extra: Revisar que decenso de gradiente tienda a la
configuración más cercana en el espacio de parámetros.
"""
import roboticstoolbox as rtb
import numpy as np

from experimRobo import denorm_q

# Método de pseudoinverso de jacobiano
def ikine_pi_jacob(q_start, x_target, fkine, jacob,
                eta=0.01, min_error=0.01, max_iters=1000):
    q = q_start
    for i in range(max_iters):
        delta_x = x_target - fkine(q)
        error = np.linalg.norm(delta_x)
        if error < min_error:
            break
        #error = dif**2
        #d_error = 2*dif * jacob(q_start)
        #q_start -= eta * d_error

        pi_jacob = np.linalg.pinv(jacob(q))

        q_update = eta * np.dot(pi_jacob, delta_x)
        q += q_update

    return q


# Método de transpuesta de jacobiano


if __name__ == "__main__":

    robot = rtb.models.DH.Cobra600()

    q0 = denorm_q(robot, [0.1, 0.1, 0.1, 0.1])

    q_target = denorm_q(robot, [0.1, 0.4, 0.2, 0.3])
    x_target = robot.fkine(q_target).t

    q_inv = ikine_pi_jacob(q0, x_target,
                        lambda q: robot.fkine(q).t, 
                        lambda q: robot.jacob0(q)[:3],
                        min_error=0)

    print(f"""
        Initial x: {robot.fkine(q0).t}
        Initial q: {q0}
        Requested x: {x_target}
        Found q: {q_inv}
        Reached x: {robot.fkine(q_inv).t}""")