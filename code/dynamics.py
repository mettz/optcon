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

# number_of_states = 6
number_of_states = 3
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

    # x, y, psi, V, beta, psi_dot = xx
    V, beta, psi_dot = xx
    delta, Fx = uu

    # Defintion of lateral forces
    Fyf = mu * Fzf * (delta - (V * np.sin(beta) + a * psi_dot) / (V * np.cos(beta)))  # mu * Fzf * Bf
    Fyr = mu * Fzr * (-(V * np.sin(beta) - b * psi_dot) / (V * np.cos(beta)))  # mu * Fzr * Br

    xx_plus = np.zeros((number_of_states,))

    # AUTONOMOUS CAR MODEL
    # xx = (x, y, psi, V, beta, psi_dot)
    # uu = (delta, Fx)

    # Euler discretization of the continuous time model
    # x_plus = x + dt * (V * np.cos(beta) * np.cos(psi) - V * np.sin(beta) * np.sin(psi))

    # y_plus = y + dt * (V * np.cos(beta) * np.sin(psi) + V * np.sin(beta) * np.cos(psi))

    # psi_plus = psi + dt * psi_dot

    V_plus = V + dt * ((1 / mass) * (Fyr * np.sin(beta) + Fx * np.cos(beta - delta) + Fyf * np.sin(beta - delta)))

    beta_plus = beta + dt * (1 / (mass * V) * (Fyr * np.cos(beta) + Fyf * np.cos(beta - delta) - Fx * np.sin(beta - delta)) - psi_dot)

    psi_dot_plus = psi_dot + dt * (1 / Iz) * ((Fx * np.sin(delta) + Fyf * np.cos(delta)) * a - Fyr * b)

    # xx_plus = np.array([x_plus, y_plus, psi_plus, V_plus, beta_plus, psi_dot_plus]).squeeze()
    xx_plus = np.array([V_plus, beta_plus, psi_dot_plus]).squeeze()

    # nabla_x = np.array(
    #     [
    #         1,
    #         0,
    #         dt * (-V * np.sin(beta) * np.cos(psi) - V * np.sin(psi) * np.cos(beta)),
    #         dt * (-np.sin(beta) * np.sin(psi) + np.cos(beta) * np.cos(psi)),
    #         dt * (-V * np.sin(beta) * np.cos(psi) - V * np.sin(psi) * np.cos(beta)),
    #         0,
    #     ]
    # )

    # nabla_y = np.array(
    #     [
    #         0,
    #         1,
    #         dt * (-V * np.sin(beta) * np.sin(psi) + V * np.cos(beta) * np.cos(psi)),
    #         dt * (np.sin(beta) * np.cos(psi) + np.sin(psi) * np.cos(beta)),
    #         dt * (-V * np.sin(beta) * np.sin(psi) + V * np.cos(beta) * np.cos(psi)),
    #         0,
    #     ]
    # )

    # nabla_psi = np.array([0, 0, 1, 0, 0, dt])

    nabla_V_eq = np.array(
        [
            # 0,
            # 0,
            # 0,
            dt*(Fzf*mu*(-np.sin(beta)/(V*np.cos(beta)) + (V*np.sin(beta) + a*psi_dot)/(V**2*np.cos(beta)))*np.sin(beta - delta) - Fzr*mu*np.sin(beta)**2/(V*np.cos(beta)) - Fzr*mu*(-V*np.sin(beta) + b*psi_dot)*np.sin(beta)/(V**2*np.cos(beta)))/mass + 1,
            dt*(-Fx*np.sin(beta - delta) + Fzf*mu*(-1 - (V*np.sin(beta) + a*psi_dot)*np.sin(beta)/(V*np.cos(beta)**2))*np.sin(beta - delta) + Fzf*mu*(delta - (V*np.sin(beta) + a*psi_dot)/(V*np.cos(beta)))*np.cos(beta - delta) - Fzr*mu*np.sin(beta) + Fzr*mu*(-V*np.sin(beta) + b*psi_dot)*np.sin(beta)**2/(V*np.cos(beta)**2) + Fzr*mu*(-V*np.sin(beta) + b*psi_dot)/V)/mass,
            dt*(-Fzf*a*mu*np.sin(beta - delta)/(V*np.cos(beta)) + Fzr*b*mu*np.sin(beta)/(V*np.cos(beta)))/mass,
        ]
    )

    nabla_beta_eq = np.array(
        [
            # 0,
            # 0,
            # 0,
            dt*((Fzf*mu*(-np.sin(beta)/(V*np.cos(beta)) + (V*np.sin(beta) + a*psi_dot)/(V**2*np.cos(beta)))*np.cos(beta - delta) - Fzr*mu*np.sin(beta)/V - Fzr*mu*(-V*np.sin(beta) + b*psi_dot)/V**2)/(V*mass) - (-Fx*np.sin(beta - delta) + Fzf*mu*(delta - (V*np.sin(beta) + a*psi_dot)/(V*np.cos(beta)))*np.cos(beta - delta) + Fzr*mu*(-V*np.sin(beta) + b*psi_dot)/V)/(V**2*mass)),
            1 + dt*(-Fx*np.cos(beta - delta) + Fzf*mu*(-1 - (V*np.sin(beta) + a*psi_dot)*np.sin(beta)/(V*np.cos(beta)**2))*np.cos(beta - delta) - Fzf*mu*(delta - (V*np.sin(beta) + a*psi_dot)/(V*np.cos(beta)))*np.sin(beta - delta) - Fzr*mu*np.cos(beta))/(V*mass),
            dt*(-1 + (-Fzf*a*mu*np.cos(beta - delta)/(V*np.cos(beta)) + Fzr*b*mu/V)/(V*mass)),
        ]
    )

    nabla_psi_dot_eq = np.array(
        [
            # 0,
            # 0,
            # 0,
            dt*(Fzf*a*mu*(-np.sin(beta)/(V*np.cos(beta)) + (V*np.sin(beta) + a*psi_dot)/(V**2*np.cos(beta)))*np.cos(delta) + Fzr*b*mu*np.sin(beta)/(V*np.cos(beta)) + Fzr*b*mu*(-V*np.sin(beta) + b*psi_dot)/(V**2*np.cos(beta)))/Iz,
            dt*(Fzf*a*mu*(-1 - (V*np.sin(beta) + a*psi_dot)*np.sin(beta)/(V*np.cos(beta)**2))*np.cos(delta) + Fzr*b*mu - Fzr*b*mu*(-V*np.sin(beta) + b*psi_dot)*np.sin(beta)/(V*np.cos(beta)**2))/Iz,
            1 + dt*(-Fzf*a**2*mu*np.cos(delta)/(V*np.cos(beta)) - Fzr*b**2*mu/(V*np.cos(beta)))/Iz,
        ]
    )

    # fx = np.array([nabla_x, nabla_y, nabla_psi, nabla_V, nabla_beta, nabla_psi_dot])
    fx_eq = np.array([nabla_V_eq, nabla_beta_eq, nabla_psi_dot_eq])

    nabla_delta_eq = np.array(
        [
            # 0,
            # 0,
            # 0,
            dt*(Fx*np.sin(beta - delta) - Fzf*mu*(delta - (V*np.sin(beta) + a*psi_dot)/(V*np.cos(beta)))*np.cos(beta - delta) + Fzf*mu*np.sin(beta - delta))/mass,
            dt*(Fx*np.cos(beta - delta) + Fzf*mu*(delta - (V*np.sin(beta) + a*psi_dot)/(V*np.cos(beta)))*np.sin(beta - delta) + Fzf*mu*np.cos(beta - delta))/(V*mass),
            a*dt*(Fx*np.cos(delta) - Fzf*mu*(delta - (V*np.sin(beta) + a*psi_dot)/(V*np.cos(beta)))*np.sin(delta) + Fzf*mu*np.cos(delta))/Iz
        ]
    )

    nabla_Fx_eq = np.array([
        # 0, 
        # 0, 
        # 0,
        dt*np.cos(beta - delta)/mass,
        -dt*np.sin(beta - delta)/(V*mass),
        a*dt*np.sin(delta)/Iz
        ])

    fu_eq = np.array([nabla_delta_eq, nabla_Fx_eq])

    return xx_plus, fx_eq, fu_eq


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