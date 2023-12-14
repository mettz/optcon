from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt

# Definition of parameters
mass = 1480  # Kg
Iz = 1950  # Kgm^2
a = 1.421  # m
b = 1.029  # m
mu = 1  # nodim
g = 9.81  # m/s^2

# Definition of discretization step
dt = 1e-3

def nonlinear_system_discretized(variables):
    # We impose beta and V to compute the equilibrium point
    beta = 20.0
    V = 1.0

    # Define your nonlinear system of equations with inputs
    _, _, psi, psi_dot, delta, Fx = variables

    Fzf = (mass * g * b) / (a + b)
    Fzr = (mass * g * a) / (a + b)

    Bf = delta - (V * np.sin(beta) + a * psi_dot) / (V * np.cos(beta))
    Br = -(V * np.sin(beta) - b * psi_dot) / (V * np.cos(beta))

    # Defintion of lateral forces
    Fyf = mu * Fzf * Bf
    Fyr = mu * Fzr * Br

    eq0 = dt * (V * np.cos(beta) * np.cos(psi) - V * np.sin(beta) * np.sin(psi))
    eq1 = dt * (V * np.cos(beta) * np.sin(psi) + V * np.sin(beta) * np.cos(psi))
    eq2 = dt * psi_dot
    eq3 = dt * ((1 / mass) * (
        Fyr * np.sin(beta) + Fx * np.cos(beta - delta) + Fyf * np.sin(beta - delta)
    ))
    eq4 = dt * (
        1
        / (mass * V)
        * (Fyr * np.cos(beta) + Fyf * np.cos(beta - delta) - Fx * np.sin(beta - delta))
        - psi_dot
    )
    eq5 = dt * ((1 / Iz) * ((Fx * np.sin(delta) + Fyf * np.cos(delta)) * a - Fyr * b))

    return [eq0, eq1, eq2, eq3, eq4, eq5]

# def nonlinear_system_continuos(variables):
#     # We impose beta and V to compute the equilibrium point
#     beta = 20.0
#     V = 1.0

#     # Define your nonlinear system of equations with inputs
#     _, _, psi, psi_dot, delta, Fx = variables

#     Fzf = (mass * g * b) / (a + b)
#     Fzr = (mass * g * a) / (a + b)

#     Bf = delta - (V * np.sin(beta) + a * psi_dot) / (V * np.cos(beta))
#     Br = -(V * np.sin(beta) - b * psi_dot) / (V * np.cos(beta))

#     # Defintion of lateral forces
#     Fyf = mu * Fzf * Bf
#     Fyr = mu * Fzr * Br

#     eq0 = (V * np.cos(beta) * np.cos(psi) - V * np.sin(beta) * np.sin(psi))
#     eq1 = (V * np.cos(beta) * np.sin(psi) + V * np.sin(beta) * np.cos(psi))
#     eq2 = psi_dot
#     eq3 = ((1 / mass) * (
#         Fyr * np.sin(beta) + Fx * np.cos(beta - delta) + Fyf * np.sin(beta - delta)
#     ))
#     eq4 = (
#         1
#         / (mass * V)
#         * (Fyr * np.cos(beta) + Fyf * np.cos(beta - delta) - Fx * np.sin(beta - delta))
#         - psi_dot
#     )
#     eq5 = ((1 / Iz) * ((Fx * np.sin(delta) + Fyf * np.cos(delta)) * a - Fyr * b))

#     return [eq0, eq1, eq2, eq3, eq4, eq5]

# Function to find equilibrium point for a given input
def find_equilibrium_point(f):
    # Initial random guess for the equilibrium point
    # initial_guess = np.random.randint(10, size=8)
    initial_guess = np.array([0, 0, 0, 0, 0, 0]) # [x:(x, y, psi,beta , psi_dot), u:(delta, Fx)] #Notarstefano ha detto di imporre beta e V
    print("Initial guess:", initial_guess)
    equilibrium_point = fsolve(f, initial_guess)
    return equilibrium_point

    # x_dot = x + dt * (V * np.cos(beta) * np.cos(psi) - V * np.sin(beta) * np.sin(psi))
    # x_k+1 - x_k = 0 = dt * (V * np.cos(beta) * np.cos(psi) - V * np.sin(beta) * np.sin(psi))
    
    # 0 = dt * (V * np.cos(beta) * np.cos(psi) - V * np.sin(beta) * np.sin(psi))=r(x)

if __name__ == "__main__":
    equilibrium_point_discretized = find_equilibrium_point(nonlinear_system_discretized)
    print("Equilibrium Point Discretized: ", equilibrium_point_discretized)

    equilibrium_point_continuos = find_equilibrium_point(nonlinear_system_continuos)
    print("Equilibrium Point Continuos: ", equilibrium_point_continuos)