import numpy as np

from dynamics import *
from scipy.optimize import fsolve

def nonlinear_system_discretized(variables, V_des, beta_des):
    '''Function that defines the nonlinear system of equations to find the equilibrium point'''
    # Definition of the system variables for the equilibrium finding
    psi_dot, delta, Fx = variables

    # Definition of the vertical forces on the front and rear wheel
    Fzf = (mass * g * b) / (a + b)
    Fzr = (mass * g * a) / (a + b)
    '''Queste si possono sicuramente recuperare dal file dynamics.py, tanto sono costanti'''
    
    # Definition of the lateral forces
    Fyf = mu * Fzf * (delta - (V * np.sin(beta) + a * psi_dot) / (V * np.cos(beta)))  # mu * Fzf * Bf
    Fyr = mu * Fzr * (-(V * np.sin(beta) - b * psi_dot) / (V * np.cos(beta)))  # mu * Fzr * Br
    '''Queste penso si possano recuperare dal file dynamics.py, però sono valori dinamici. Quindi bisogna sincerarsi 
    che in dynamics.py i valori siano aggiornati di pari passo con equilibrium.py'''

    # Definition of the system of equations to find the roots
    eq1 = dt * ((1 / mass) * (Fyr * np.sin(beta_des) + Fx * np.cos(beta_des - delta) + Fyf * np.sin(beta_des - delta)))
    eq2 = dt * (1 / (mass * V_des) * (Fyr * np.cos(beta_des) + Fyf * np.cos(beta_des - delta) - Fx * np.sin(beta_des - delta)) - psi_dot)
    eq3 = dt * ((1 / Iz) * ((Fx * np.sin(delta) + Fyf * np.cos(delta)) * a - Fyr * b))

    return [eq1, eq2, eq3]


def find_equilibrium_point(f, **kwargs):
    '''Function that finds the equilibrium point of the system for a given input'''
    # Initial random guess for the equilibrium point
    initial_guess = kwargs.get("initial_guess") # [x:(psi_dot), u:(delta, Fx)]
    args = kwargs.get("args")
    print("Initial guess:", initial_guess)

    # Find the equilibrium point
    equilibrium_point = fsolve(f, initial_guess, args=args)
    
    # Mathemathical check of the equilibrium point
    print(f"Is close?  {np.isclose(nonlinear_system_discretized(equilibrium_point, args[0], args[1]), np.zeros(3))}")
    return equilibrium_point
