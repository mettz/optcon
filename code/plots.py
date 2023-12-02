import numpy as np
import matplotlib.pyplot as plt
import dynamics

#Debug Luca
#===== Plots of the equilibrium points using equilibria as initial conditions =========
def plot_with_equilibria(equilibrium_point, V_des, beta_des):
    # Plot the equilibrium points
        psi_dot, delta, Fx = equilibrium_point
        xx = np.array([0, 0, 0, V_des, beta_des, psi_dot])
        # Devo dare in ingresso anche x, y e psi corrispondenti al punto di equilibrio atrimenti è normale vedere un transitorio iniziale
        uu = np.array([delta, Fx])

        steps = np.linspace(0, 100, 100000)
        trajectory_xx = np.zeros((len(steps), len(xx)))

        for i in range(len(steps)):
            xx_plus = dynamics.dynamics(xx, uu)
            trajectory_xx[i, :] = xx_plus
            xx = xx_plus

        plt.figure()
        plt.clf()
        plt.plot(steps, trajectory_xx[:, 0], label="x")
        plt.plot(steps, trajectory_xx[:, 1], label="y")
        plt.plot(steps, trajectory_xx[:, 3], label="V")
        plt.plot(steps, trajectory_xx[:, 4], label="beta")
        plt.plot(steps, trajectory_xx[:, 5], label="psi_dot")
        plt.xlabel("Time")
        plt.ylabel("State variables")
        plt.title("State variables at the equilibrium")
        plt.ylim([-1000, 1000])
        plt.grid()
        plt.legend()
        plt.show()

        plt.figure()
        plt.clf()
        plt.plot(steps, trajectory_xx[:, 2], label="psi")
        plt.xlabel("Time")
        plt.ylabel("State variable: psi")
        plt.title("State variables at the equilibrium")
        plt.grid()
        plt.legend()
        plt.show()

def plot_equilibria(equilibrium_point, V_des, beta_des):
    # Plot the equilibrium points
    psi_dot, delta, Fx = equilibrium_point
    xx = np.array([0, 0, 0, V_des, beta_des, psi_dot])
    uu = np.array([delta, Fx])

    steps, trajectory_xx, trajectory_uu = dynamics.trajectory(100, xx, uu)

    plt.figure()
    plt.clf()
    plt.plot(steps, trajectory_xx[:, 0], label="x")
    plt.plot(steps, trajectory_xx[:, 1], label="y")
    plt.plot(steps, trajectory_xx[:, 3], label="V")
    plt.plot(steps, trajectory_xx[:, 4], label="beta")
    plt.plot(steps, trajectory_xx[:, 5], label="psi_dot")
    plt.xlabel("Time")
    plt.ylabel("State variables")
    plt.title("State variables behaviour at the equilibrium")
    plt.ylim([-1000, 1000])
    plt.grid()
    plt.legend()
    plt.show()

    plt.figure()
    plt.clf()
    plt.plot(steps, trajectory_xx[:, 2], label="psi")
    plt.xlabel("Time")
    plt.ylabel("State variable: psi")
    plt.title("State variables behaviour at the equilibrium")
    plt.grid()
    plt.legend()
    plt.show()

    plt.figure()
    plt.clf()
    plt.plot(steps, trajectory_uu[:, 0], label="delta")
    plt.plot(steps, trajectory_uu[:, 1], label="Fx")
    plt.title("Input variables behaviour at the equilibrium")
    plt.grid()
    plt.legend()
    plt.show()

#==================== Plot of the dynamics =====================================
def dynamics_plot(delta, Fx):
    # Considering constant inputs
    delta = 45.0
    Fx = 1.0
    uu = np.array([delta, Fx])
    xx = np.array([0, 1.0, 0, 5.0, 1.0, 1])
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
    plt.show()