import numpy as np
from sympy import symbols, diff

x, y, psi, V, beta, psi_dot = symbols("x y psi V beta psi_dot")

dt = 1e-3  # discretization step
mass = 1480  # Kg
Iz = 1950  # Kgm^2
a = 1.421  # m
b = 1.029  # m
mu = 1  # nodim
g = 9.81  # m/s^2

number_of_states = 6
number_of_inputs = 2


def dynamics(xx, uu):
    """
    Dynamics of a discrete-time mass spring damper system

    Args
        - xx \in \R^6 state at time t
        - uu \in \R^2 input at time t

    Return
        - next state xx_{t+1}
    """

    xx = xx.squeeze()
    uu = uu.squeeze()

    # Definition of the vertical forces on the front and rear wheel
    Fzf = (mass * g * b) / (a + b)
    Fzr = (mass * g * a) / (a + b)

    x, y, psi, V, beta, psi_dot = xx
    delta, Fx = uu

    # Defintion of lateral forces
    Fyf = mu * Fzf * (delta - (V * np.sin(beta) + a * psi_dot) / (V * np.cos(beta)))  # mu * Fzf * Bf
    Fyr = mu * Fzr * (-(V * np.sin(beta) - b * psi_dot) / (V * np.cos(beta)))  # mu * Fzr * Br

    xx_plus = np.zeros((number_of_states,))

    # AUTONOMOUS CAR MODEL
    # xx = (x, y, psi, V, beta, psi_dot)
    # uu = (delta, Fx)

    # Euler discretization of the continuous time model
    x_plus = x + dt * (V * np.cos(beta) * np.cos(psi) - V * np.sin(beta) * np.sin(psi))

    y_plus = y + dt * (V * np.cos(beta) * np.sin(psi) + V * np.sin(beta) * np.cos(psi))

    psi_plus = psi + dt * psi_dot

    V_plus = V + dt * ((1 / mass) * (Fyr * np.sin(beta) + Fx * np.cos(beta - delta) + Fyf * np.sin(beta - delta)))

    beta_plus = beta + dt * (1 / (mass * V) * (Fyr * np.cos(beta) + Fyf * np.cos(beta - delta) - Fx * np.sin(beta - delta)) - psi_dot)

    psi_dot_plus = psi_dot + dt * (1 / Iz) * ((Fx * np.sin(delta) + Fyf * np.cos(delta)) * a - Fyr * b)

    xx_plus = np.array([x_plus, y_plus, psi_plus, V_plus, beta_plus, psi_dot_plus]).squeeze()

    nabla_x = np.array(
        [
            1,
            0,
            dt * (-V * np.sin(beta) * np.cos(psi) - V * np.sin(psi) * np.cos(beta)),
            dt * (-np.sin(beta) * np.sin(psi) + np.cos(beta) * np.cos(psi)),
            dt * (-V * np.sin(beta) * np.cos(psi) - V * np.sin(psi) * np.cos(beta)),
            0,
        ]
    )

    nabla_y = np.array(
        [
            0,
            1,
            dt * (-V * np.sin(beta) * np.sin(psi) + V * np.cos(beta) * np.cos(psi)),
            dt * (np.sin(beta) * np.cos(psi) + np.sin(psi) * np.cos(beta)),
            dt * (-V * np.sin(beta) * np.sin(psi) + V * np.cos(beta) * np.cos(psi)),
            0,
        ]
    )

    nabla_psi = np.array([0, 0, 1, 0, 0, dt])

    nabla_V = np.array(
        [
            0,
            0,
            0,
            (
                dt
                * (
                    b * g * mass * mu * (-np.sin(beta) / (V * np.cos(beta)) + (V * np.sin(beta) + a * psi_dot) / (V**2 * np.cos(beta))) * np.sin(beta - delta) / (a + b)
                    - a * g * mass * mu * np.sin(beta) ** 2 / (V * (a + b) * np.cos(beta))
                    - a * g * mass * mu * (-V * np.sin(beta) + b * psi_dot) * np.sin(beta) / (V**2 * (a + b) * np.cos(beta))
                )
                / mass
                + 1
            ),
            (
                dt
                * (
                    -Fx * np.sin(beta - delta)
                    - a * g * mass * mu * np.sin(beta) / (a + b)
                    + b * g * mass * mu * (-1 - (V * np.sin(beta) + a * psi_dot) * np.sin(beta) / (V * np.cos(beta) ** 2)) * np.sin(beta - delta) / (a + b)
                    + b * g * mass * mu * (delta - (V * np.sin(beta) + a * psi_dot) / (V * np.cos(beta))) * np.cos(beta - delta) / (a + b)
                    + a * g * mass * mu * (-V * np.sin(beta) + b * psi_dot) * np.sin(beta) ** 2 / (V * (a + b) * np.cos(beta) ** 2)
                    + a * g * mass * mu * (-V * np.sin(beta) + b * psi_dot) / (V * (a + b))
                )
                / mass
            ),
            dt * (a * b * g * mass * mu * np.sin(beta) / (V * (a + b) * np.cos(beta)) - a * b * g * mass * mu * np.sin(beta - delta) / (V * (a + b) * np.cos(beta))) / mass,
        ]
    )

    nabla_beta = np.array(
        [
            0,
            0,
            0,
            dt
            * (
                (
                    b * g * mass * mu * (-np.sin(beta) / (V * np.cos(beta)) + (V * np.sin(beta) + a * psi_dot) / (V**2 * np.cos(beta))) * np.cos(beta - delta) / (a + b)
                    - a * g * mass * mu * np.sin(beta) / (V * (a + b))
                    - a * g * mass * mu * (-V * np.sin(beta) + b * psi_dot) / (V**2 * (a + b))
                )
                / (V * mass)
                - (
                    -Fx * np.sin(beta - delta)
                    + b * g * mass * mu * (delta - (V * np.sin(beta) + a * psi_dot) / (V * np.cos(beta))) * np.cos(beta - delta) / (a + b)
                    + a * g * mass * mu * (-V * np.sin(beta) + b * psi_dot) / (V * (a + b))
                )
                / (V**2 * mass)
            ),
            1
            + dt
            * (
                -Fx * np.cos(beta - delta)
                - a * g * mass * mu * np.cos(beta) / (a + b)
                + b * g * mass * mu * (-1 - (V * np.sin(beta) + a * psi_dot) * np.sin(beta) / (V * np.cos(beta) ** 2)) * np.cos(beta - delta) / (a + b)
                - b * g * mass * mu * (delta - (V * np.sin(beta) + a * psi_dot) / (V * np.cos(beta))) * np.sin(beta - delta) / (a + b)
            )
            / (V * mass),
            dt * (-1 + (a * b * g * mass * mu / (V * (a + b)) - a * b * g * mass * mu * np.cos(beta - delta) / (V * (a + b) * np.cos(beta))) / (V * mass)),
        ]
    )

    nabla_psi_dot = np.array(
        [
            0,
            0,
            0,
            (
                dt
                * (
                    a * b * g * mass * mu * (-np.sin(beta) / (V * np.cos(beta)) + (V * np.sin(beta) + a * psi_dot) / (V**2 * np.cos(beta))) * np.cos(delta) / (a + b)
                    + a * b * g * mass * mu * np.sin(beta) / (V * (a + b) * np.cos(beta))
                    + a * b * g * mass * mu * (-V * np.sin(beta) + b * psi_dot) / (V**2 * (a + b) * np.cos(beta))
                )
                / Iz
            ),
            (
                dt
                * (
                    a * b * g * mass * mu * (-1 - (V * np.sin(beta) + a * psi_dot) * np.sin(beta) / (V * np.cos(beta) ** 2)) * np.cos(delta) / (a + b)
                    + a * b * g * mass * mu / (a + b)
                    - a * b * g * mass * mu * (-V * np.sin(beta) + b * psi_dot) * np.sin(beta) / (V * (a + b) * np.cos(beta) ** 2)
                )
                / Iz
            ),
            1 + dt * (-(a**2) * b * g * mass * mu * np.cos(delta) / (V * (a + b) * np.cos(beta)) - a * b**2 * g * mass * mu / (V * (a + b) * np.cos(beta))) / Iz,
        ]
    )

    fx = np.array([nabla_x, nabla_y, nabla_psi, nabla_V, nabla_beta, nabla_psi_dot])

    nabla_delta = np.array(
        [
            0,
            0,
            0,
            dt
            * (
                Fx * np.sin(beta - delta)
                - b * g * mass * mu * (delta - (V * np.sin(beta) + a * psi_dot) / (V * np.cos(beta))) * np.cos(beta - delta) / (a + b)
                + b * g * mass * mu * np.sin(beta - delta) / (a + b)
            )
            / mass,
            dt
            * (
                Fx * np.cos(beta - delta)
                + b * g * mass * mu * (delta - (V * np.sin(beta) + a * psi_dot) / (V * np.cos(beta))) * np.sin(beta - delta) / (a + b)
                + b * g * mass * mu * np.cos(beta - delta) / (a + b)
            )
            / (V * mass),
            a
            * dt
            * (
                Fx * np.cos(delta)
                - b * g * mass * mu * (delta - (V * np.sin(beta) + a * psi_dot) / (V * np.cos(beta))) * np.sin(delta) / (a + b)
                + b * g * mass * mu * np.cos(delta) / (a + b)
            )
            / Iz,
        ]
    )

    nabla_Fx = np.array([0, 0, 0, dt * np.cos(beta - delta) / mass, -dt * np.sin(beta - delta) / (V * mass), a * dt * np.sin(delta) / Iz])

    fu = np.array([nabla_delta, nabla_Fx])

    return xx_plus, fx, fu


def trajectory(points, xx, uu):
    steps = np.linspace(0, points, points * 1000)
    trajectory_xx = np.zeros((len(steps), len(xx)))
    trajectory_uu = np.zeros((len(steps), len(uu)))

    for i in range(len(steps)):
        xx_plus, _, _ = dynamics(xx, uu)
        trajectory_xx[i, :] = xx_plus
        xx = xx_plus
        trajectory_uu[i, :] = uu

    return steps, trajectory_xx, trajectory_uu
