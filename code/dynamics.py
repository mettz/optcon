import numpy as np

from constants import DT, NUMBER_OF_STATES

# Definition of the vehicle parameters
mass = 1480  # Kg
Iz = 1950  # Kgm^2
a = 1.421  # m
b = 1.029  # m
mu = 1  # nodim
g = 9.81  # m/s^2

# Definition of the vertical forces on the front and rear wheel
Fzf = (mass * g * b) / (a + b)
Fzr = (mass * g * a) / (a + b)


def Fyf(xx, uu):
    V, beta, psi_dot = xx
    delta, _ = uu

    Bf = delta - (V * np.sin(beta) + a * psi_dot) / (V * np.cos(beta))
    return mu * Fzf * Bf


def Fyr(xx, _):
    V, beta, psi_dot = xx

    Br = -(V * np.sin(beta) - b * psi_dot) / (V * np.cos(beta))
    return mu * Fzr * Br


def V(xx, uu):
    _, beta, _ = xx
    delta, Fx = uu

    return (1 / mass) * (Fyr(xx, uu) * np.sin(beta) + Fx * np.cos(beta - delta) + Fyf(xx, uu) * np.sin(beta - delta))


def beta(xx, uu):
    V, beta, psi_dot = xx
    delta, Fx = uu

    return (1 / (mass * V)) * (Fyr(xx, uu) * np.cos(beta) + Fyf(xx, uu) * np.cos(beta - delta) - Fx * np.sin(beta - delta)) - psi_dot


def psi_dot(xx, uu):
    delta, Fx = uu

    return (1 / Iz) * ((Fx * np.sin(delta) + Fyf(xx, uu) * np.cos(delta)) * a - Fyr(xx, uu) * b)


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

    V, beta, psi_dot = xx
    delta, Fx = uu

    # Definition of the lateral forces
    Fyf = mu * Fzf * (delta - (V * np.sin(beta) + a * psi_dot) / (V * np.cos(beta)))  # mu * Fzf * Bf
    Fyr = mu * Fzr * (-(V * np.sin(beta) - b * psi_dot) / (V * np.cos(beta)))  # mu * Fzr * Br

    xx_plus = np.zeros((NUMBER_OF_STATES,))

    V_plus = V + DT * ((1 / mass) * (Fyr * np.sin(beta) + Fx * np.cos(beta - delta) + Fyf * np.sin(beta - delta)))

    beta_plus = beta + DT * (1 / (mass * V) * (Fyr * np.cos(beta) + Fyf * np.cos(beta - delta) - Fx * np.sin(beta - delta)) - psi_dot)

    psi_dot_plus = psi_dot + DT * (1 / Iz) * ((Fx * np.sin(delta) + Fyf * np.cos(delta)) * a - Fyr * b)

    xx_plus = np.array(
        [
            V_plus,
            beta_plus,
            psi_dot_plus,
        ]
    ).squeeze()

    nabla_V = np.array(
        [
            DT
            * (
                Fzf * mu * (-np.sin(beta) / (V * np.cos(beta)) + (V * np.sin(beta) + a * psi_dot) / (V**2 * np.cos(beta))) * np.sin(beta - delta)
                - Fzr * mu * np.sin(beta) ** 2 / (V * np.cos(beta))
                - Fzr * mu * (-V * np.sin(beta) + b * psi_dot) * np.sin(beta) / (V**2 * np.cos(beta))
            )
            / mass
            + 1,
            DT
            * (
                -Fx * np.sin(beta - delta)
                + Fzf * mu * (-1 - (V * np.sin(beta) + a * psi_dot) * np.sin(beta) / (V * np.cos(beta) ** 2)) * np.sin(beta - delta)
                + Fzf * mu * (delta - (V * np.sin(beta) + a * psi_dot) / (V * np.cos(beta))) * np.cos(beta - delta)
                - Fzr * mu * np.sin(beta)
                + Fzr * mu * (-V * np.sin(beta) + b * psi_dot) * np.sin(beta) ** 2 / (V * np.cos(beta) ** 2)
                + Fzr * mu * (-V * np.sin(beta) + b * psi_dot) / V
            )
            / mass,
            DT * (-Fzf * a * mu * np.sin(beta - delta) / (V * np.cos(beta)) + Fzr * b * mu * np.sin(beta) / (V * np.cos(beta))) / mass,
        ]
    )

    nabla_beta = np.array(
        [
            DT
            * (
                (
                    Fzf * mu * (-np.sin(beta) / (V * np.cos(beta)) + (V * np.sin(beta) + a * psi_dot) / (V**2 * np.cos(beta))) * np.cos(beta - delta)
                    - Fzr * mu * np.sin(beta) / V
                    - Fzr * mu * (-V * np.sin(beta) + b * psi_dot) / V**2
                )
                / (V * mass)
                - (
                    -Fx * np.sin(beta - delta)
                    + Fzf * mu * (delta - (V * np.sin(beta) + a * psi_dot) / (V * np.cos(beta))) * np.cos(beta - delta)
                    + Fzr * mu * (-V * np.sin(beta) + b * psi_dot) / V
                )
                / (V**2 * mass)
            ),
            1
            + DT
            * (
                -Fx * np.cos(beta - delta)
                + Fzf * mu * (-1 - (V * np.sin(beta) + a * psi_dot) * np.sin(beta) / (V * np.cos(beta) ** 2)) * np.cos(beta - delta)
                - Fzf * mu * (delta - (V * np.sin(beta) + a * psi_dot) / (V * np.cos(beta))) * np.sin(beta - delta)
                - Fzr * mu * np.cos(beta)
            )
            / (V * mass),
            DT * (-1 + (-Fzf * a * mu * np.cos(beta - delta) / (V * np.cos(beta)) + Fzr * b * mu / V) / (V * mass)),
        ]
    )

    nabla_psi_dot = np.array(
        [
            DT
            * (
                Fzf * a * mu * (-np.sin(beta) / (V * np.cos(beta)) + (V * np.sin(beta) + a * psi_dot) / (V**2 * np.cos(beta))) * np.cos(delta)
                + Fzr * b * mu * np.sin(beta) / (V * np.cos(beta))
                + Fzr * b * mu * (-V * np.sin(beta) + b * psi_dot) / (V**2 * np.cos(beta))
            )
            / Iz,
            DT
            * (
                Fzf * a * mu * (-1 - (V * np.sin(beta) + a * psi_dot) * np.sin(beta) / (V * np.cos(beta) ** 2)) * np.cos(delta)
                + Fzr * b * mu
                - Fzr * b * mu * (-V * np.sin(beta) + b * psi_dot) * np.sin(beta) / (V * np.cos(beta) ** 2)
            )
            / Iz,
            1 + DT * (-Fzf * a**2 * mu * np.cos(delta) / (V * np.cos(beta)) - Fzr * b**2 * mu / (V * np.cos(beta))) / Iz,
        ]
    )

    fx = np.array(
        [
            nabla_V,
            nabla_beta,
            nabla_psi_dot,
        ]
    )

    nabla_delta = np.array(
        [
            DT
            * (Fx * np.sin(beta - delta) - Fzf * mu * (delta - (V * np.sin(beta) + a * psi_dot) / (V * np.cos(beta))) * np.cos(beta - delta) + Fzf * mu * np.sin(beta - delta))
            / mass,
            DT
            * (Fx * np.cos(beta - delta) + Fzf * mu * (delta - (V * np.sin(beta) + a * psi_dot) / (V * np.cos(beta))) * np.sin(beta - delta) + Fzf * mu * np.cos(beta - delta))
            / (V * mass),
            a * DT * (Fx * np.cos(delta) - Fzf * mu * (delta - (V * np.sin(beta) + a * psi_dot) / (V * np.cos(beta))) * np.sin(delta) + Fzf * mu * np.cos(delta)) / Iz,
        ]
    )

    nabla_Fx = np.array(
        [
            DT * np.cos(beta - delta) / mass,
            -DT * np.sin(beta - delta) / (V * mass),
            a * DT * np.sin(delta) / Iz,
        ]
    )

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

def get_X_Y(xx):
    V, beta, psi_dot = xx
    X_pos_plus = np.zeros((NUMBER_OF_STATES,))
    Y_pos_plus = np.zeros((NUMBER_OF_STATES,))

    #Integrate psi_dot to get psi
    psi = np.zeros((NUMBER_OF_STATES,))
    for tt in range(len(xx)):
        psi[tt] = np.trapz(psi_dot[0:tt], dx=DT)
    
    for tt in range(len(xx)-1):
        X_pos_plus[tt+1] = X_pos_plus[tt] + DT * (V[tt] * np.cos(beta[tt]) * np.cos(psi[tt]) - V[tt] * np.sin(beta[tt]) * np.sin(psi[tt]))
        Y_pos_plus[tt+1] = X_pos_plus[tt] + DT * (V[tt] * np.cos(beta[tt]) * np.sin(psi[tt]) + V[tt] * np.sin(beta[tt]) * np.cos(psi[tt]))

    return X_pos_plus, Y_pos_plus