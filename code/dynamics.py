import numpy as np

# Definition of parameters
dt = 1e-3  # discretization step
mass = 1480  # Kg
Iz = 1950  # Kgm^2
a = 1.421  # m
b = 1.029  # m
mu = 1  # nodim
g = 9.81  # m/s^2


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

    number_of_states = 6
    number_of_inputs = 2

    # Definition of the vertical forces on the front and rear wheel
    Fzf = (mass * g * b) / (a + b)
    Fzr = (mass * g * a) / (a + b)

    x, y, psi, V, beta, psi_dot = xx
    delta, Fx = uu

    # Defintion of front and rear slideslip angles
    Bf = delta - (V * np.sin(beta) + a * psi_dot) / (V * np.cos(beta))
    Br = -(V * np.sin(beta) - b * psi_dot) / (V * np.cos(beta))

    # Defintion of lateral forces
    Fyf = mu * Fzf * (delta - (V * np.sin(beta) + a * psi_dot) / (V * np.cos(beta)))  # mu * Fzf * Bf
    Fyr = mu * Fzr * (-(V * np.sin(beta) - b * psi_dot) / (V * np.cos(beta)))  # mu * Fzr * Br

    xx_plus = np.zeros((number_of_states,))

    # AUTONOMOUS CAR MODEL
    # x = (x, y, psi, V, beta, psi_dot)
    # x[0] = x
    # x[1] = y
    # x[2] = psi
    # x[3] = V
    # x[4] = beta
    # x[5] = psi_dot

    # u = (delta, Fx)
    # u[0] = delta
    # u[1] = Fx

    # Euler discretization (we use it since the equations become simpler)
    # x_dot = V * cos(beta) * cos(psi) - V * sin(beta) * sin(psi) # Continuous time
    x_dot = x + dt * (V * np.cos(beta) * np.cos(psi) - V * np.sin(beta) * np.sin(psi)) # Discrete time

    # y_dot = V * cos(beta) * sin(psi) + V * sin(beta) * cos(psi) # Continuous time
    y_dot = y + dt * (V * np.cos(beta) * np.sin(psi) + V * np.sin(beta) * np.cos(psi)) # Discrete time

    # psi_dot = psi_dot # Continuous time
    psi_dot = psi + dt * psi_dot # Discrete time

    # mass * V_dot = Fyr * sin(beta) + Fx * cos(beta - delta) + Fyf * sin(beta - delta) # Continuous time
    V_dot = V + dt * ((1 / mass) * (Fyr * np.sin(beta) + Fx * np.cos(beta - delta) + Fyf * np.sin(beta - delta))) # Discrete time

    # beta_dot = 1 / (mass * V) * (Fy,r * cos(beta) + Fy,f * cos(beta - delta) - Fx * sin(beta - delta)) - psi_dot # Continuous time
    beta_dot = beta + dt * (1 / (mass * V) * ((Fyr * np.cos(beta) + Fyf * np.cos(beta - delta) - Fx * np.sin(beta - delta)) - psi_dot)) # Discrete time

    # Iz * psi_dot_dot = (Fx * sin(delta) + Fy,f * cos(delta)) * a - Fy,r * b # Continuous time
    psi_dot_dot = psi_dot + dt * ((1 / Iz) * ((Fx * np.sin(delta) + Fyf * np.cos(delta)) * a - Fyr * b)) # Discrete time

    # Computation of the gradient of the discretized dynamics equations
    nabla0 = np.array( # Derivate of x_dot with respect to (x,y,psi,V,beta,psi_dot)
        [ 
            1,
            0,
            -dt * V * (np.cos(beta) * np.sin(psi) + np.sin(beta) * np.cos(psi)),
            dt * (np.cos(beta) * np.cos(psi) - np.sin(beta) * np.sin(psi)),
            -dt * V (np.sin(beta) * np.cos(psi) - np.cos(beta) * np.sin(psi)),
            0,
        ] #OK
    )
    nabla1 = np.array( # Derivate of y_dot with respect to (x,y,psi,V,beta,psi_dot)
        [ 
            0,
            1,
            dt * V * (np.cos(beta) * np.cos(psi) - np.sin(beta) * np.sin(psi)),
            dt * (np.cos(beta) * np.sin(psi) + np.sin(beta) * np.cos(psi)),
            -dt * V * (np.sin(beta) * np.sin(psi) - V * np.cos(beta) * np.cos(psi)),
            0,
        ] #OK
    )
    nabla2 = np.array([0, 0, 1, 0, 0, dt]) # Derivate of psi_dot with respect to (x,y,psi,V,beta,psi_dot) #OK 
    nabla3 = np.array( # Derivate of V_dot with respect to (x,y,psi,V,beta,psi_dot)
        [ #CONTROLLARE E AGGIUNGERE I DT
            0,
            0,
            0,
            0,
            (1 / mass) * (Fyr * np.cos(beta) - Fx * np.sin(beta - delta) + Fyf * np.cos(beta - delta)),
            0,
        ]
    )
    nabla4 = np.array( # Derivate of beta_dot with respect to (x,y,psi,V,beta,psi_dot)
        [ #CONTROLLARE E AGGIUNGERE I DT
            0,
            0,
            0,
            -1 / (mass * (V**2)) * dt * (Fyr * np.cos(beta) + Fyf * np.cos(beta - delta) - Fx * np.sin(beta - delta)),
            1 + dt * (1 / mass * V) * Fyr * (-np.sin(beta)) - Fyf * np.sin(beta - delta) - Fx * np.cos(beta - delta),
            -1,
        ]
    )
    nabla5 = np.array( # Derivate of psi_dot_dot with respect to (x,y,psi,V,beta,psi_dot) 
        [ #CONTROLLARE E AGGIUNGERE I DT
            0,
            0,
            0,
            psi_dot * a * (np.cos(beta) * a + 1) * b  ,
            0,
            1
         ]
        ) 

    # Code for debugging and testing
    # print('nabla0', nabla0.shape)
    # prova = np.array([nabla0, nabla1, nabla2, nabla3, nabla4, nabla5])

    # print('prova.shape', prova.shape)
    # print('prova', prova)

    fx = np.column_stack((nabla0, nabla1, nabla2, nabla3, nabla4, nabla5))
    # print('fx', fx.shape)
    # print('fx', fx)

    fu = np.array(
        [
            [
                0,
                0,
                0,
                dt * (1 / mass) * (Fx * np.sin(beta - delta) - Fyf * np.cos(beta - delta)),
                dt * (1 / mass * V) * (Fyf * np.sin(beta - delta) + Fx * np.cos(beta - delta)),
                dt * 1 / Iz * (Fx * np.cos(delta) - Fyf * np.sin(delta)) * a,
            ],
            [
                0,
                0,
                0,
                dt * 1 / mass * np.cos(beta - delta),
                dt * (1 / (mass * V) * (-np.sin(beta - delta))),
                dt * 1 / Iz * np.sin(delta) * a,
            ],
        ]
    )

    xx_plus = np.array([x_dot, y_dot, psi_dot, V_dot, beta_dot, psi_dot_dot]).squeeze()

    return xx_plus, fx, fu
