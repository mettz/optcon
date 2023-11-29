from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
from dynamics import dynamics
from dynamics_continuous import dynamics_continuous

# Definition of parameters
mass = 1480  # Kg
Iz = 1950  # Kgm^2
a = 1.421  # m
b = 1.029  # m
mu = 1  # nodim
g = 9.81  # m/s^2

# Definition of discretization step
dt = 1e-3

# We impose beta and V to compute the equilibrium point
beta_des = 20.0
V_des = 1.0

# Definition of the state initial conditions
x_des = 10.0
y_des = 10.0
psi_des = 45.0
psi_dot_init = 0.0


def nonlinear_system_discretized(variables):
    # Define your nonlinear system of equations with inputs
    psi_dot, delta, Fx = variables

    Fzf = (mass * g * b) / (a + b)
    Fzr = (mass * g * a) / (a + b)

    Bf = delta - (V_des * np.sin(beta_des) + a * psi_dot) / (V_des * np.cos(beta_des))
    Br = -(V_des * np.sin(beta_des) - b * psi_dot) / (V_des * np.cos(beta_des))

    # Defintion of lateral forces
    Fyf = mu * Fzf * Bf
    Fyr = mu * Fzr * Br

    eq3 = dt * ((1 / mass) * (Fyr * np.sin(beta_des) + Fx * np.cos(beta_des - delta) + Fyf * np.sin(beta_des - delta)))
    eq4 = dt * (1 / (mass * V_des) * (Fyr * np.cos(beta_des) + Fyf * np.cos(beta_des - delta) - Fx * np.sin(beta_des - delta)) - psi_dot)
    eq5 = dt * ((1 / Iz) * ((Fx * np.sin(delta) + Fyf * np.cos(delta)) * a - Fyr * b))

    return [eq3, eq4, eq5]


def nonlinear_system_continuous(variables):
    # Define your nonlinear system of equations with inputs
    psi_dot, delta, Fx = variables

    Fzf = (mass * g * b) / (a + b)
    Fzr = (mass * g * a) / (a + b)

    Bf = delta - (V_des * np.sin(beta_des) + a * psi_dot) / (V_des * np.cos(beta_des))
    Br = -(V_des * np.sin(beta_des) - b * psi_dot) / (V_des * np.cos(beta_des))

    # Defintion of lateral forces
    Fyf = mu * Fzf * Bf
    Fyr = mu * Fzr * Br

    eq3 = (1 / mass) * (Fyr * np.sin(beta_des) + Fx * np.cos(beta_des - delta) + Fyf * np.sin(beta_des - delta))
    eq4 = 1 / (mass * V_des) * (Fyr * np.cos(beta_des) + Fyf * np.cos(beta_des - delta) - Fx * np.sin(beta_des - delta)) - psi_dot
    eq5 = (1 / Iz) * ((Fx * np.sin(delta) + Fyf * np.cos(delta)) * a - Fyr * b)

    return [eq3, eq4, eq5]


# Function to find equilibrium point for a given input
def find_equilibrium_point(f):
    # Initial random guess for the equilibrium point
    # initial_guess = np.random.randint(10, size=8)
    initial_guess = np.array([0, 0, 0])  # [x:(psi_dot), u:(delta, Fx)]
    print("Initial guess:", initial_guess)
    equilibrium_point = fsolve(f, initial_guess)
    return equilibrium_point

    # x_dot = x + dt * (V * np.cos(beta) * np.cos(psi) - V * np.sin(beta) * np.sin(psi))
    # x_k+1 - x_k = 0 = dt * (V * np.cos(beta) * np.cos(psi) - V * np.sin(beta) * np.sin(psi))

    # 0 = dt * (V * np.cos(beta) * np.cos(psi) - V * np.sin(beta) * np.sin(psi))=r(x)


def plot_equilibria(equilibrium_point):
    # Plot the equilibrium points
    psi_dot, delta, Fx = equilibrium_point
    xx = np.array([x_des, y_des, psi_des, V_des, beta_des, psi_dot])
    uu = np.array([delta, Fx])

    steps = np.linspace(0, 50, 5000)
    trajectory = np.zeros((len(steps), len(xx)))

    xx2 = np.array([0, 0, 0, 2, 10, 5])
    uu2 = np.array([10, 1])

    for i in range(len(steps)):
        xx_plus = dynamics(xx2, uu2)
        #xx_plus = dynamics(xx, uu)
        trajectory[i, :] = xx_plus
        xx = xx_plus

    plt.figure()
    plt.clf()
    plt.plot(steps, trajectory[:, 0], label="x")
    plt.plot(steps, trajectory[:, 1], label="y")
    plt.plot(steps, trajectory[:, 2], label="psi")
    plt.plot(steps, trajectory[:, 3], label="V")
    plt.plot(steps, trajectory[:, 4], label="beta")
    plt.plot(steps, trajectory[:, 5], label="psi_dot")
    plt.xlabel("Time")
    plt.ylabel("State variables")
    plt.ylim([-100, 100])
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    equilibrium_point_discretized = find_equilibrium_point(nonlinear_system_discretized)
    print("Equilibrium Point Discretized: ", equilibrium_point_discretized)
    print("Diff: ", np.isclose(nonlinear_system_discretized(equilibrium_point_discretized), np.zeros(3)))

    equilibrium_point_continuous = find_equilibrium_point(nonlinear_system_continuous)
    print("Equilibrium Point Continuous: ", equilibrium_point_continuous)
    print("Diff: ", np.isclose(nonlinear_system_continuous(equilibrium_point_continuous), np.zeros(3)))

    plot_equilibria(equilibrium_point_continuous)
