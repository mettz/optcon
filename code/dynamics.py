import numpy as np

# discretization step
dt = 1e-3

def dynamics(xx,uu):
    """
    Dynamics of a discrete-time mass spring damper system

    Args
        - xx \in \R^6 state at time t
        - uu \in \R^2 input at time t

    Return 
        - next state xx_{t+1}
    """
    
    xx = xx.squeeze() #Perchè dovremo farlo?
    uu = uu.squeeze()

    number_of_states = 6
    number_of_inputs = 2
         
    #Definition of parameters
    mass = 1480 #Kg
    Iz = 1950 #Kgm^2
    a = 1.421 #m
    b = 1.029 #m
    mu = 1 #nodim
    g = 9.81 #m/s^2

    #Defintion of lateral forces
    Fyf = mu * Fzf * Bf
    Fyr = mu * Fzr * Br
    
    #Defintion of front and rear slideslip angles
    Bf = delta - (x[3] * np.sin(B) + a * u[4]) / (V * np.cos(B))
    Br = - (x[3] * np.sin(B) - b * u[5]) / (x[3] * np.cos(B))
    
    #Definition of the vertical forces on the front and rear wheel
    Fzf = (mass * g * b) / (a + b)
    Fzr = (mass * g * a) / (a + b)

    xx_plus = np.zeros((number_of_states, ))

    # Autonomous car model
    # x = (x, y, psi, V, beta, psi_dot), u = (delta, Fx)
    # x[0] = x
    # x[1] = y
    # x[2] = psi
    # x[3] = V
    # x[4] = beta
    # x[5] = psi_dot
    # u[0] = delta
    # u[1] = Fx

    # Euler discretization (we use it since the equations become simpler)
    # x_dot = V * cos(beta) * cos(psi) - V * sin(beta) * sin(psi)
    xx_plus[0] = xx[0] + dt * (xx[3] * np.cos(xx[4]) * np.cos(xx[2]) - xx[3] * np.sin(xx[4]) * np.sin(xx[2]))
    
    # y_dot = V * cos(beta) * sin(psi) + V * sin(beta) * cos(psi)
    xx_plus[1] = xx[1] + dt * (xx[3] * np.cos(xx[4]) * np.sin(xx[2]) + xx[3] * np.sin(xx[4]) * np.sin(xx[2]))
    
    # m * V_dot = Fy,r * sin(beta) + Fx * cos(beta - delta) + Fy,f * sin(beta - delta)
    xx_plus[2] = xx[2] + dt * ((1 / mass) * Fyr * np.sin(xx[4]) + uu[1] * np.cos(xx[4] - uu[0]) + Fyf * np.sin(xx[4] - uu[0]))
    
    # psi_dot = x[5]
    xx_plus[3] = xx[3] + dt * xx[5]
    
    # beta_dot = 1 / (m * V) * (Fy,r * cos(beta) + Fy,f * cos(beta - delta) - Fx * sin(beta - delta)) - psi_dot
    xx_plus[4] = xx[4] + dt * (1 / mass * xx[2]) * (Fyr * np.cos(xx[4]) + Fyf * np.cos(xx[4] - uu[0]) + Fyf * np.sin(xx[4] - uu[0]))
    
    # Iz * psi_dot_dot = (Fx * sin(delta) + Fy,f * cos(delta)) * a - Fy,r * b
    xx_plus[5] = xx[5] + dt * ((1 / Iz) * (uu[1] * np.sin(uu[0]) + Fyf * np.cos(uu[0])) * a - Fyr * b)

    # Gradient

    # Mass-spring-damper cart
    # x1_d = x2
    # x2_d = -kspring / mm x1 - bb/mm x2 + 1/mm u

    # Here we compute the matrix A_t and B_t of the dynamics (in this case the dynamic is linear
    # so there is no need to compute the linearization)
    # fx = np.array([[1, -dt * kspring / mm],
    #     [dt, 1 + dt * (- bb / mm)]])

    # fu = np.array([[0, dt * 1 / mm]])

    return xx_plus #, fx, fu