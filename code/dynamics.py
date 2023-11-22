import numpy as np

#Definition of parameters
dt = 1e-3 # discretization step
mass = 1480 #Kg
Iz = 1950 #Kgm^2
a = 1.421 #m
b = 1.029 #m
mu = 1 #nodim
g = 9.81 #m/s^2

def dynamics(xx,uu):
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

    #Definition of the vertical forces on the front and rear wheel
    Fzf = (mass * g * b) / (a + b)
    Fzr = (mass * g * a) / (a + b)

    #Defintion of front and rear slideslip angles
    Bf = uu[0] - (xx[3] * np.sin(xx[4]) + a * xx[5]) / (xx[3] * np.cos(xx[4]))
    Br = - (xx[3] * np.sin(xx[4]) - b * xx[5]) / (xx[3] * np.cos(xx[4])) 

    #Defintion of lateral forces
    Fyf = mu * Fzf * Bf
    Fyr = mu * Fzr * Br

    xx_plus = np.zeros((number_of_states, ))

    # AUTONOMOUS CAR MODEL
    # x = (x, y, psi, V, beta, psi_dot)
    # x[0] = x
    # x[1] = y
    # x[2] = psi
    # x[3] = V
    # x[4] = beta
    # x[5] = psi_dot

    #u = (delta, Fx)
    # u[0] = delta
    # u[1] = Fx

    # Euler discretization (we use it since the equations become simpler)
    # x_dot = V * cos(beta) * cos(psi) - V * sin(beta) * sin(psi)
    xx_plus[0] = xx[0] + dt * ( xx[3] * np.cos(xx[4]) * np.cos(xx[2]) - xx[3] * np.sin(xx[4]) * np.sin(xx[2]) )
    
    # y_dot = V * cos(beta) * sin(psi) + V * sin(beta) * cos(psi)
    xx_plus[1] = xx[1] + dt * ( xx[3] * np.cos(xx[4]) * np.sin(xx[2]) + xx[3] * np.sin(xx[4]) * np.cos(xx[2]) )
    
    # psi_dot = x[5]
    xx_plus[2] = xx[2] + dt * xx[5]
    
    # m * V_dot = Fy,r * sin(beta) + Fx * cos(beta - delta) + Fy,f * sin(beta - delta)
    xx_plus[3] = xx[3] + dt * ( (1/mass) * (Fyr * np.sin(xx[4]) + uu[1] * np.cos(xx[4] - uu[0]) + Fyf * np.sin(xx[4] - uu[0])) )

    # beta_dot = 1 / (m * V) * (Fy,r * cos(beta) + Fy,f * cos(beta - delta) - Fx * sin(beta - delta)) - psi_dot
    xx_plus[4] = xx[4] + dt * ( 1/(mass * xx[3]) * ( Fyr * np.cos(xx[4]) + Fyf * np.cos(xx[4] - uu[0]) - uu[1] * np.sin(xx[4] - uu[0])) - xx[5])
    
    # Iz * psi_dot_dot = (Fx * sin(delta) + Fy,f * cos(delta)) * a - Fy,r * b
    xx_plus[5] = xx[5] + dt * ((1/Iz) * ((uu[1] * np.sin(uu[0]) + Fyf * np.cos(uu[0])) * a - Fyr * b))

    # Computation of the gradient of the dynamics equations
    nabla0 = np.array([1 , 0, dt * (-xx[3]*np.cos(xx[4])*np.sin(xx[2]) - xx[3]*np.sin(xx[4])*np.cos(xx[2])), dt * (np.cos(xx[4])*np.cos(xx[2]) - np.sin(xx[4])*np.sin(xx[2])), dt * (-xx[3]*np.sin(xx[4])*np.cos(xx[2]) - xx[3]*np.cos(xx[4])*np.sin(xx[2])), 0])
    nabla1 = np.array([0, 1, dt * (xx[3]*np.cos(xx[4])*np.cos(xx[2]) - xx[3]*np.sin(xx[4])*np.sin(xx[2])), dt * (np.cos(xx[4])*np.sin(xx[2])+np.sin(xx[4])*np.cos(xx[2])), dt * (-xx[3]*np.sin(xx[4])*np.sin(xx[2])+xx[3]*np.cos(xx[4])*np.cos(xx[2])), 0])
    nabla2 = np.array([0, 0, 1, 0, 0, dt])
    nabla3 = np.array([0, 0, 0, 1, dt * (1/mass) * (Fyr * np.cos(xx[4]) - uu[1] * np.sin(xx[4]-uu[0]) + Fyf * np.cos(xx[4] - uu[0])), 0])
    nabla4 = np.array([0, 0, 0, -1/(mass * (xx[3] ** 2)) * dt * (Fyr * np.cos(xx[4]) + Fyf * np.cos(xx[4] - uu[0]) - uu[1] * np.sin(xx[4] - uu[0])), 1 + dt * (1 / mass*xx[3]) * Fyr * (-np.sin(xx[4])) - Fyf * np.sin(xx[4] - uu[0]) - uu[1] * np.cos(xx[4] - uu[0]), -dt]) 
    nabla5 = np.array([0, 0, 0, 0, 0, 1])
    
    #Code for debugging and testing
    #print('nabla0', nabla0.shape)
    #prova = np.array([nabla0, nabla1, nabla2, nabla3, nabla4, nabla5])

    #print('prova.shape', prova.shape)
    #print('prova', prova)
        
    fx = np.column_stack((nabla0, nabla1, nabla2, nabla3, nabla4, nabla5))
    #print('fx', fx.shape)
    #print('fx', fx)
    
    fu = np.array([ [0, 0, 0, dt * (1/mass) * ( uu[1] * np.sin(xx[4]-uu[0]) - Fyf * np.cos(xx[4]-uu[0]) ), dt * (1/mass*xx[3]) * ( Fyf * np.sin(xx[4]-uu[0]) + uu[1] * np.cos(xx[4]-uu[0]) ), dt * 1/Iz * (uu[1] * np.cos(uu[0]) - Fyf * np.sin(uu[0]))*a],
                    [0, 0, 0, dt * 1/mass * np.cos(xx[4]-uu[0]), dt * (1/(mass*xx[3]) * (-np.sin(xx[4]-uu[0])) ), dt * 1/Iz * np.sin(uu[0]) * a] ])

    xx_plus.squeeze()
    
    return xx_plus, fx, fu