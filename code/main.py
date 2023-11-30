from scipy.optimize import fsolve
from scipy.optimize import minimize
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from dynamics import dynamics
from dynamics_continuous import dynamics_continuous
from autograd import jacobian
from autograd import hessian

# import cost functions
import cost as cost

from scipy.linalg import solve_discrete_are
from scipy.integrate import solve_ivp

# Allow Ctrl-C to work despite plotting
import signal

signal.signal(signal.SIGINT, signal.SIG_DFL)

# Definition of parameters
mass = 1480  # Kg
Iz = 1950  # Kgm^2
a = 1.421  # m
b = 1.029  # m
mu = 1  # nodim
g = 9.81  # m/s^2

# Definition of discretization step
dt = 1e-3

def nonlinear_system_discretized(variables, beta_des, V_des):
    # Define your nonlinear system of equations with inputs
    psi_dot, delta, Fx = variables

    Fzf = (mass * g * b) / (a + b)
    Fzr = (mass * g * a) / (a + b)

    Bf = delta - (V_des * np.sin(beta_des) + a * psi_dot) / (V_des * np.cos(beta_des))
    Br = -(V_des * np.sin(beta_des) - b * psi_dot) / (V_des * np.cos(beta_des))

    # Defintion of lateral forces
    Fyf = mu * Fzf * Bf
    Fyr = mu * Fzr * Br

    eq1 = dt * ((1 / mass) * (Fyr * np.sin(beta_des) + Fx * np.cos(beta_des - delta) + Fyf * np.sin(beta_des - delta)))
    eq2 = dt * (1 / (mass * V_des) * (Fyr * np.cos(beta_des) + Fyf * np.cos(beta_des - delta) - Fx * np.sin(beta_des - delta)) - psi_dot)
    eq3 = dt * ((1 / Iz) * ((Fx * np.sin(delta) + Fyf * np.cos(delta)) * a - Fyr * b))

    return [eq1, eq2, eq3]


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

    eq1 = (1 / mass) * (Fyr * np.sin(beta_des) + Fx * np.cos(beta_des - delta) + Fyf * np.sin(beta_des - delta))
    eq2 = 1 / (mass * V_des) * (Fyr * np.cos(beta_des) + Fyf * np.cos(beta_des - delta) - Fx * np.sin(beta_des - delta)) - psi_dot
    eq3 = (1 / Iz) * ((Fx * np.sin(delta) + Fyf * np.cos(delta)) * a - Fyr * b)

    return [eq1, eq2, eq3]


# Function to find equilibrium point for a given input
def find_equilibrium_point(f, **kwargs):
    # Initial random guess for the equilibrium point
    # initial_guess is [x:(psi_dot), u:(delta, Fx)]
    initial_guess = kwargs.get("initial_guess")
    args = kwargs.get("args")
    print("Initial guess:", initial_guess)
    equilibrium_point = fsolve(f, initial_guess, args=args)
    return equilibrium_point

#Debug Luca
#===== Plots of the equilibrium points using equilibria as initial conditions =========
def plot_with_equilibria(equilibrium_point):
    # Plot the equilibrium points
        psi_dot, delta, Fx = equilibrium_point
        xx = np.array([0, 0, 0, V_des, beta_des, psi_dot])
        uu = np.array([delta, Fx])

        steps = np.linspace(0, 100, 100000)
        trajectory = np.zeros((len(steps), len(xx)))

        for i in range(len(steps)):
            xx_plus = dynamics(xx, uu)
            trajectory[i, :] = xx_plus
            xx = xx_plus

        plt.figure()
        plt.clf()
        plt.plot(steps, trajectory[:, 0], label="x")
        plt.plot(steps, trajectory[:, 1], label="y")
        plt.plot(steps, trajectory[:, 3], label="V")
        plt.plot(steps, trajectory[:, 4], label="beta")
        plt.plot(steps, trajectory[:, 5], label="psi_dot")
        plt.xlabel("Time")
        plt.ylabel("State variables")
        plt.title("State variables at the equilibrium")
        plt.ylim([-250, 250])
        plt.grid()
        plt.legend()
        plt.show()

        plt.figure()
        plt.clf()
        plt.plot(steps, trajectory[:, 2], label="psi")
        plt.xlabel("Time")
        plt.ylabel("State variable: psi")
        plt.title("State variables at the equilibrium")
        plt.grid()
        plt.legend()
        plt.show()

def plot_equilibria(equilibrium_point):
    # Plot the equilibrium points
    psi_dot, delta, Fx = equilibrium_point
    xx = np.array([0, 0, 0, V_des, beta_des, psi_dot])
    uu = np.array([delta, Fx])

    steps = np.linspace(0, 100, 100000)
    trajectory = np.zeros((len(steps), len(xx)))

    for i in range(len(steps)):
        xx_plus = dynamics(xx, uu)
        trajectory[i, :] = xx_plus
        xx = xx_plus

    plt.figure()
    plt.clf()
    plt.plot(steps, trajectory[:, 0], label="x")
    plt.plot(steps, trajectory[:, 1], label="y")
    plt.plot(steps, trajectory[:, 3], label="V")
    plt.plot(steps, trajectory[:, 4], label="beta")
    plt.plot(steps, trajectory[:, 5], label="psi_dot")
    plt.xlabel("Time")
    plt.ylabel("State variables")
    plt.title("State variables behaviour at the equilibrium")
    plt.ylim([-250, 250])
    plt.grid()
    plt.legend()
    plt.show()

    plt.figure()
    plt.clf()
    plt.plot(steps, trajectory[:, 2], label="psi")
    plt.xlabel("Time")
    plt.ylabel("State variable: psi")
    plt.title("State variables behaviour at the equilibrium")
    plt.grid()
    plt.legend()
    plt.show()

if __name__ == "__main__":
    beta_des = 20.0
    V_des = 1.0
    eq1 = find_equilibrium_point(nonlinear_system_discretized, initial_guess=[0, 0, 0], args=(beta_des, V_des))
    print("1st eq point: ", eq1)
    print("Diff: ", np.isclose(nonlinear_system_discretized(eq1, beta_des, V_des), np.zeros(3)))
    # Valore non corrispondente di psi_dot nel grafico ottenuto: non va bene!
    #plot_equilibria(eq1)
    plot_with_equilibria(eq1)

    beta_des = 10.0
    V_des = 1.0
    eq2 = find_equilibrium_point(nonlinear_system_discretized, initial_guess=[0, 0, 0], args=(beta_des, V_des))
    print("2nd eq point: ", eq2)
    print("Diff: ", np.isclose(nonlinear_system_discretized(eq2, beta_des, V_des), np.zeros(3)))
    #plot_equilibria(eq2)
    plot_with_equilibria(eq2)
    # Comportamento molto strano (direi sbagliato) delle variabili di stato, x instabile

    beta_des = 60.0
    V_des = -5.0
    eq3 = find_equilibrium_point(nonlinear_system_discretized, initial_guess=[0, 0, 0], args=(beta_des, V_des))
    print("3rd eq point: ", eq3)
    print("Diff: ", np.isclose(nonlinear_system_discretized(eq3, beta_des, V_des), np.zeros(3)))
    plot_equilibria(eq3)
    plot_with_equilibria(eq3)
    # Otteniamo i valori desiderati nei plot di beta_des e V_des, e viene plottato correttamente il valore di psi_dot

    #==================== Plot of the dynamics =====================================
    # Considering constant inputs
    '''delta = 45.0
    Fx = 1.0
    uu = np.array([delta, Fx])
    xx = np.array([10, 10, 20, 1.0, 0.0, 1])
    dyn = np.zeros([100000, 2])
    # dyn: x0 y0
    #      x1 y1
    #      .. ..
    #      xN yN
    # -> dyn: (N,2)

    for i in range(100000):
        xx_plus = dynamics(xx,uu)
        dyn[i, 0] = xx_plus[0]
        dyn[i, 1] = xx_plus[1]
        xx = xx_plus

    plt.figure()
    plt.clf()
    plt.plot(dyn[:,0], dyn[:,1])
    plt.xlabel("State variable: x")
    plt.ylabel("State variable: y")
    plt.title("Dynamics")
    plt.grid()
    plt.show() '''
    # Si vedono le circonference che ci diceva Bossio



'''
# Newton's method for optimal control
def newtons_method(x0, max_iter=100, tol=1e-6):
    x = x0
    for _ in range(max_iter):
        # Evaluate the dynamics, cost, and constraints at the current iterate
        dynamics_x = dynamics(x, None)  # You may need to include the current control input
        cost_x = cost(None)  # You need to include the current control input

        # Formulate the Lagrangian Hessian matrix
        lagrangian_hessian = hessian(dynamics_x)  # Replace with the actual Hessian matrix of the Lagrangian

        # Solve the linearized subproblem using the Hessian matrix
        linearized_subproblem = minimize(cost, x, method='SLSQP', jac=None, hess=lagrangian_hessian, constraints=None)

        # Update the control input
        u_new = linearized_subproblem.x

        # Check for convergence
        if np.linalg.norm(u_new - x) < tol:
            break

        x = u_new

    return x'''