import numpy as np
from scipy.optimize import fsolve

from dynamics import PARAMETERS

def nonlinear_system_discretized(variables, V_des, beta_des):
    # Define your nonlinear system of equations with inputs
    psi_dot, delta, Fx = variables

    Fzf = (PARAMETERS['mass'] * PARAMETERS['g'] * PARAMETERS['b']) / (PARAMETERS['a'] + PARAMETERS['b'])
    Fzr = (PARAMETERS['mass'] * PARAMETERS['g'] * PARAMETERS['a']) / (PARAMETERS['a'] + PARAMETERS['b'])

    Bf = delta - (V_des * np.sin(beta_des) + PARAMETERS['a'] * psi_dot) / (V_des * np.cos(beta_des))
    Br = -(V_des * np.sin(beta_des) - PARAMETERS['b'] * psi_dot) / (V_des * np.cos(beta_des))

    # Defintion of lateral forces
    Fyf = PARAMETERS['mu'] * Fzf * Bf
    Fyr = PARAMETERS['mu'] * Fzr * Br

    # Define the system of equations to find the roots
    eq1 = PARAMETERS['dt'] * ((1 / PARAMETERS['mass']) * (Fyr * np.sin(beta_des) + Fx * np.cos(beta_des - delta) + Fyf * np.sin(beta_des - delta)))
    eq2 = PARAMETERS['dt'] * (1 / (PARAMETERS['mass'] * V_des) * (Fyr * np.cos(beta_des) + Fyf * np.cos(beta_des - delta) - Fx * np.sin(beta_des - delta)) - psi_dot)
    eq3 = PARAMETERS['dt'] * ((1 / PARAMETERS['Iz']) * ((Fx * np.sin(delta) + Fyf * np.cos(delta)) * PARAMETERS['a'] - Fyr * PARAMETERS['b']))

    return [eq1, eq2, eq3]


def nonlinear_system_continuous(variables, V_des, beta_des):
    # Define your nonlinear system of equations with inputs
    psi_dot, delta, Fx = variables

    Fzf = (PARAMETERS['mass'] * PARAMETERS['g'] * PARAMETERS['b']) / (PARAMETERS['a'] + PARAMETERS['b'])
    Fzr = (PARAMETERS['mass'] * PARAMETERS['g'] * PARAMETERS['a']) / (PARAMETERS['a'] + PARAMETERS['b'])

    Bf = delta - (V_des * np.sin(beta_des) + PARAMETERS['a'] * psi_dot) / (V_des * np.cos(beta_des))
    Br = -(V_des * np.sin(beta_des) - PARAMETERS['b'] * psi_dot) / (V_des * np.cos(beta_des))

    # Defintion of lateral forces
    Fyf = PARAMETERS['mu'] * Fzf * Bf
    Fyr = PARAMETERS['mu'] * Fzr * Br

    eq1 = (1 / PARAMETERS['mass']) * (Fyr * np.sin(beta_des) + Fx * np.cos(beta_des - delta) + Fyf * np.sin(beta_des - delta))
    eq2 = 1 / (PARAMETERS['mass'] * V_des) * (Fyr * np.cos(beta_des) + Fyf * np.cos(beta_des - delta) - Fx * np.sin(beta_des - delta)) - psi_dot
    eq3 = (1 / PARAMETERS['Iz']) * ((Fx * np.sin(delta) + Fyf * np.cos(delta)) * PARAMETERS['a'] - Fyr * PARAMETERS['b'])

    return [eq1, eq2, eq3]


# Function to find equilibrium point for a given input
def find_equilibrium_point(f, **kwargs):
    # Initial random guess for the equilibrium point
    # initial_guess is [x:(psi_dot), u:(delta, Fx)]
    initial_guess = kwargs.get("initial_guess")
    args = kwargs.get("args")
    print("Initial guess:", initial_guess)
    equilibrium_point = fsolve(f, initial_guess, args=args)
    print(f"Is close?  {np.isclose(nonlinear_system_discretized(equilibrium_point, args[0], args[1]), np.zeros(3))}")
    return equilibrium_point