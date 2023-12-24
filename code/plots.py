import matplotlib.pyplot as plt
import numpy as np

import constants
import dynamics as dyn

# ===== Plots of the equilibrium points using equilibria as initial conditions =========
def verify_equilibria(equilibrium_state, equilibrium_input, V_des, beta_des):
    # Plot of the system dynamics starting by the equilibrium points
    V_des, beta_des, psi_dot = equilibrium_state
    xx = np.array([V_des, beta_des, psi_dot])
    delta, Fx = equilibrium_input
    uu = np.array([delta, Fx])

    steps = np.linspace(0, 100, 100000)
    trajectory_xx = np.zeros((len(steps), len(xx)))

    for i in range(len(steps)):
        xx_plus = dyn.dynamics(xx, uu)[0]
        trajectory_xx[i, :] = xx_plus
        xx = xx_plus

    plt.figure()
    plt.clf()
    plt.plot(steps, trajectory_xx[:, 0], label="V")
    plt.plot(steps, trajectory_xx[:, 1], label="beta")
    plt.plot(steps, trajectory_xx[:, 2], label="psi_dot")
    plt.xlabel("Time")
    plt.ylabel("State variables")
    plt.title("State variables at the equilibrium")
    plt.ylim([-1000, 1000])
    plt.grid()
    plt.legend()
    plt.show()


def plot_equilibria(equilibrium_point, V_des, beta_des):
    # Plot the equilibrium points
    psi_dot, delta, Fx = equilibrium_point
    xx = np.array([0, 0, 0, V_des, beta_des, psi_dot])
    uu = np.array([delta, Fx])

    steps, trajectory_xx, trajectory_uu = dyn.trajectory(100, xx, uu)

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


########################  Plot of the dynamics  #############################
def dynamics_plot(xx_init, uu):
    # Considering constant inputs
    delta, Fx = uu
    V_init, beta_init, psi_dot_init = xx_init

    uu = np.array([delta, Fx])
    xx = np.array([V_init, beta_init, psi_dot_init])

    steps = 10000
    xx_yy = np.zeros([steps, 2])
    # dyn: x0 y0
    #      x1 y1
    #      .. ..
    #      xN yN
    # -> dyn: (N,2)

    for i in range(steps):
        xx_plus = dyn(xx, uu)
        xx_yy[i, 0] = xx_plus[0]
        xx_yy[i, 1] = xx_plus[1]
        xx = xx_plus

    plt.figure()
    plt.clf()
    plt.plot(dyn[:, 0], dyn[:, 1])
    plt.xlabel("State variable: x")
    plt.ylabel("State variable: y")
    plt.title("Dynamics")
    plt.grid()
    plt.show()

################# Plot of the derivatives over the trajectory #################
def derivatives_plot(xx_traj, uu_traj):
    LinPoint = int (constants.TT / 2)
    xx_plus = np.zeros((constants.NUMBER_OF_STATES, constants.TT))
    gradient_taylor_timed = np.zeros((constants.NUMBER_OF_STATES, constants.TT))
    fx = np.zeros((constants.NUMBER_OF_STATES, constants.NUMBER_OF_STATES, constants.TT))
    xx_plus_taylor = np.zeros((constants.NUMBER_OF_STATES, constants.TT))
    fu = np.zeros((constants.NUMBER_OF_INPUTS, constants.NUMBER_OF_STATES, constants.TT))
    
    for time in range(constants.TT):
        xx_plus[:, time], fx[:, :, time], fu[:, :, time] = dyn.dynamics(xx_traj[:, time], uu_traj[:, time])

    for time in range(constants.TT):
        gradient_taylor_timed[:, time] = xx_plus[:, time] + fx[:, :, time].T @ (xx_plus[:, time] + - xx_plus[:, LinPoint])
        xx_plus_taylor[:, time] = xx_traj[:, LinPoint] + (time - float(LinPoint)) * (xx_plus[:, LinPoint] - xx_traj[:, LinPoint])
    
    span = np.linspace((LinPoint - 10), (LinPoint + 10), 20)

    fig, axs = plt.subplots(constants.NUMBER_OF_STATES, 1, sharex=True, figsize=(10, 6))
    for idx in range(constants.NUMBER_OF_STATES):
        axs[idx].plot(span, xx_plus_taylor[idx, (LinPoint-10):(LinPoint+10)],'k', label=f'taylor_x{idx} in {LinPoint}')
        axs[idx].plot(range(constants.TT), xx_traj[idx, :], 'r--', label=f'x{idx}')
    plt.show()
############################
# Gradient method plots
############################
def armijo_plot(stepsize_0, stepsizes, costs_armijo, descent_arm, JJ, kk, cc, ns, ni, TT, x0, uu, deltau, dyn, cst, xx_ref, uu_ref):
    ############################
    # Armijo plot
    ############################
    steps = np.linspace(0, stepsize_0, int(2e1))
    costs = np.zeros(len(steps))

    for ii in range(len(steps)):
        step = steps[ii]

        # temp solution update

        xx_temp = np.zeros((ns, TT))
        uu_temp = np.zeros((ni, TT))

        xx_temp[:, 0] = x0

        for tt in range(TT - 1):
            uu_temp[:, tt] = uu[:, tt, kk] + step * deltau[:, tt, kk]
            xx_temp[:, tt + 1] = dyn.dynamics(xx_temp[:, tt], uu_temp[:, tt])[0]

        # temp cost calculation
        JJ_temp = 0

        for tt in range(TT - 1):
            temp_cost = cst.stagecost(xx_temp[:, tt], uu_temp[:, tt], xx_ref[:, tt], uu_ref[:, tt])[0]
            JJ_temp += temp_cost

        temp_cost = cst.termcost(xx_temp[:, -1], xx_ref[:, -1])[0]
        JJ_temp += temp_cost

        costs[ii] = np.min([JJ_temp, 100 * JJ[kk]])

        plt.figure(1)
        plt.clf()

        plt.plot(steps, costs, color="g", label="$J(\\mathbf{u}^k - stepsize*d^k)$")
        plt.plot(
            steps,
            JJ[kk] + descent_arm[kk] * steps,
            color="r",
            label="$J(\\mathbf{u}^k) - stepsize*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$",
        )
        # plt.plot(steps, JJ[kk] - descent[kk]*steps, color='r', label='$J(\\mathbf{u}^k) - stepsize*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$')
        plt.plot(
            steps,
            JJ[kk] + cc * descent_arm[kk] * steps,
            color="g",
            linestyle="dashed",
            label="$J(\\mathbf{u}^k) - stepsize*c*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$",
        )

        # plt.scatter(stepsizes, costs_armijo, marker="*")  # plot the tested stepsize

        plt.grid()
        plt.xlabel("stepsize")
        plt.legend()
        plt.draw()

        plt.show()


def gradient_method_plots(xx_ref, uu_ref, max_iters, xx_star, uu_star, descent, JJ, TT, tf, ni, ns):
    # cost and descent

    plt.figure("descent direction")
    plt.plot(np.arange(max_iters), descent[:max_iters])
    plt.xlabel("$k$")
    plt.ylabel("||$\\nabla J(\\mathbf{u}^k)||$")
    plt.yscale("log")
    plt.grid()
    plt.show(block=False)

    plt.figure("cost")
    plt.plot(np.arange(max_iters), JJ[:max_iters])
    plt.xlabel("$k$")
    plt.ylabel("$J(\\mathbf{u}^k)$")
    plt.yscale("log")
    plt.grid()
    plt.show(block=False)

    # optimal trajectory
    tt_hor = np.linspace(0, tf, TT)
    fig, axs = plt.subplots(ns + ni, 1, sharex="all")

    axs[0].plot(tt_hor, xx_star[0, :], linewidth=2)
    axs[0].plot(tt_hor, xx_ref[0, :], "g--", linewidth=2)
    axs[0].grid()
    axs[0].set_ylabel("$x_1$")

    axs[1].plot(tt_hor, xx_star[1, :], linewidth=2)
    axs[1].plot(tt_hor, xx_ref[1, :], "g--", linewidth=2)
    axs[1].grid()
    axs[1].set_ylabel("$x_2$")

    axs[2].plot(tt_hor, uu_star[0, :], "r", linewidth=2)
    axs[2].plot(tt_hor, uu_ref[0, :], "r--", linewidth=2)
    axs[2].grid()
    axs[2].set_ylabel("$u$")
    axs[2].set_xlabel("time")

    plt.show()
