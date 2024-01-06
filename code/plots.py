import matplotlib.pyplot as plt
import numpy as np

import constants
import dynamics as dyn
from equilibrium import Equilibrium


class Plotter:
    def __init__(self, **opts) -> None:
        self.quiet = opts.get("quiet", False)
        self.show_ref_curves_plots = opts.get("show_ref_curves_plots", False)
        self.show_verify_equilibria_plots = opts.get("show_verify_equilibria", False)
        self.show_derivatives_plots = opts.get("show_derivatives_plots", False)

    def _multiplot(self, *curves, **kwargs) -> None:
        if len(curves) == 0:
            raise ValueError("No curves to plot")

        title = kwargs.get("title", None)
        ylabels = kwargs.get("ylabels", None)
        legend = kwargs.get("legend", None)
        has_legend = legend is not None

        naxes, horizon = curves[0].shape
        time = np.arange(horizon)
        fig, axes = plt.subplots(naxes, 1, sharex="all")

        for axis in range(naxes):
            axes[axis].grid()
            axes[axis].set_xlim([-1, horizon])

            for idx, curve in enumerate(curves):
                label = has_legend and legend[idx]
                axes[axis].plot(time, curve[axis, :], linewidth=2, label=label)

            if has_legend:
                axes[axis].legend()

            if ylabels is not None:
                axes[axis].set_ylabel(f"${ylabels[axis]}$")

        axes[-1].set_xlabel("time")
        fig.align_ylabels(axes)
        if title is not None:
            plt.suptitle(title)
        plt.show()

    def reference_curves(self, xx_ref, uu_ref, **kwargs) -> None:
        if not self.show_ref_curves_plots:
            return

        curve = kwargs.get("curve", None)

        title = f"States reference curves {'' if curve is None else f'({curve})'}"
        self._multiplot(xx_ref, title=title, ylabels=constants.STATES)

        title = f"Inputs reference curves {'' if curve is None else f'({curve})'}"
        self._multiplot(uu_ref, title=title, ylabels=constants.INPUTS)

    # ===== Plots of the equilibrium points using equilibria as initial conditions =========
    def verify_equilibria(self, eq: Equilibrium, **kwargs) -> None:
        if not self.show_verify_equilibria_plots:
            return

        xx = np.array([eq.V, eq.beta, eq.psi_dot])
        uu = np.array([eq.delta, eq.Fx])

        horizon = kwargs.get("horizon", constants.TT)
        time = np.arange(horizon)
        xx_plus = np.zeros((constants.NUMBER_OF_STATES, horizon))

        xx_plus[:, 0] = xx
        for tt in range(horizon - 1):
            xx_plus[:, tt + 1] = dyn.dynamics(xx_plus[:, tt], uu)[0]

        plt.figure()
        plt.clf()
        plt.plot(time, xx_plus[0, :], label=f"${constants.STATES[0]}$")
        plt.plot(time, xx_plus[1, :], label=f"${constants.STATES[1]}$")
        plt.plot(time, xx_plus[2, :], label=f"${constants.STATES[2]}$")
        plt.xlabel("Time")
        plt.ylabel("State variables")
        plt.xlim([-1, horizon])
        plt.title("State variables at the equilibrium")
        plt.grid()
        plt.legend()
        plt.show()

    ################# Plot of the derivatives over the trajectory #################
    def derivatives_plot(self, **kwargs):
        if not self.show_derivatives_plots:
            return

        horizon = kwargs.get("horizon", constants.TT)

        xx_traj = np.ones((constants.NUMBER_OF_STATES, constants.TT))
        uu_traj = np.ones((constants.NUMBER_OF_INPUTS, constants.TT))
        for tt in range(horizon - 1):
            xx_traj[:, tt + 1] = dyn.dynamics(xx_traj[:, tt], uu_traj[:, tt])[0]

        lin_point = int(horizon / 20)
        xx_plus = np.zeros((constants.NUMBER_OF_STATES, horizon))
        gradient_taylor_timed = np.zeros((constants.NUMBER_OF_STATES, horizon))
        fx = np.zeros((constants.NUMBER_OF_STATES, constants.NUMBER_OF_STATES, horizon))
        xx_plus_taylor = np.zeros((constants.NUMBER_OF_STATES, horizon))
        fu = np.zeros((constants.NUMBER_OF_INPUTS, constants.NUMBER_OF_STATES, horizon))

        for time in range(horizon):
            xx_plus[:, time], fx[:, :, time], fu[:, :, time] = dyn.dynamics(xx_traj[:, time], uu_traj[:, time])

        for time in range(horizon):
            gradient_taylor_timed[:, time] = xx_plus[:, time] + fx[:, :, time].T @ (xx_plus[:, time] + -xx_plus[:, lin_point])
            xx_plus_taylor[:, time] = xx_traj[:, lin_point] + (time - float(lin_point)) * (xx_plus[:, lin_point] - xx_traj[:, lin_point])

        span = np.linspace((lin_point - 10), (lin_point + 10), 20)

        fig, axs = plt.subplots(constants.NUMBER_OF_STATES, 1, sharex=True, figsize=(10, 6))
        for idx in range(constants.NUMBER_OF_STATES):
            axs[idx].plot(span, xx_plus_taylor[idx, (lin_point - 10) : (lin_point + 10)], "k", label=f"taylor_x{idx} in {lin_point}")
            axs[idx].plot(range(horizon), xx_traj[idx, :], "r--", label=f"x{idx}")
        plt.show()

    def following_plots(self, xx_ref, uu_ref, xx_star, uu_star):
        if self.quiet:
            return

        self._multiplot(xx_ref, xx_star, title="State trajectory following", ylabels=constants.STATES, legend=["reference", "optimal"])
        self._multiplot(uu_ref, uu_star, title="Input trajectory following", ylabels=constants.INPUTS, legend=["reference", "optimal"])

    def lqr_plots(self, xx_star, uu_star, xx_lqr, uu_lqr):
        if self.quiet:
            return

        self._multiplot(xx_star, xx_lqr, title="State trajectory tracking (LQR)", ylabels=constants.STATES, legend=["optimal", "LQR"])
        self._multiplot(uu_star, uu_lqr, title="Input trajectory tracking (LQR)", ylabels=constants.INPUTS, legend=["optimal", "LQR"])

    def mpc_plots(self, xx_star, uu_star, xx_mpc, uu_mpc):
        if self.quiet:
            return

        self._multiplot(xx_star, xx_mpc, title="State trajectory tracking (MPC)", ylabels=constants.STATES, legend=["optimal", "MPC"])
        self._multiplot(uu_star, uu_mpc, title="Input trajectory tracking (MPC)", ylabels=constants.INPUTS, legend=["optimal", "MPC"])

    ############################
    # Gradient method plots
    ############################

    def armijo_plot(stepsize_0, stepsizes, costs_armijo, descent_arm, JJ, kk, cc, x0, uu, deltau, dyn, cst, xx_ref, uu_ref):
        ############################
        # Armijo plot
        # UPDATE: ho tolto kk da descent_arm[kk] perchè in cvxpy è stato tolto
        ############################
        steps = np.linspace(0, stepsize_0, int(4e1))
        costs = np.zeros(len(steps))

        for ii in range(len(steps)):
            step = steps[ii]

            # temp solution update

            xx_temp = np.zeros((constants.NUMBER_OF_STATES, constants.TT))
            uu_temp = np.zeros((constants.NUMBER_OF_INPUTS, constants.TT))

            xx_temp[:, 0] = x0

            for tt in range(constants.TT - 1):
                uu_temp[:, tt] = (
                    uu[:, tt, kk] + step * deltau[:, tt, kk]
                )  # prima era deltau[:, tt, kk]. Ho dovuto togliere kk perchè nel codice con CVXPY non c'è. Potrebbe non funzionare altrove
                xx_temp[:, tt + 1] = dyn.dynamics(xx_temp[:, tt], uu_temp[:, tt])[0]

            # temp cost calculation
            JJ_temp = 0

            for tt in range(constants.TT - 1):
                temp_cost = cst.stagecost(xx_temp[:, tt], uu_temp[:, tt], xx_ref[:, tt], uu_ref[:, tt])[0]
                JJ_temp += temp_cost

            temp_cost = cst.termcost(xx_temp[:, -1], xx_ref[:, -1])[0]
            JJ_temp += temp_cost

            # costs[ii] = np.min([JJ_temp, 100 * JJ[kk]]) nostro codice
            costs[ii] = JJ_temp  # Ste e Davide

        plt.figure(1)
        plt.clf()

        plt.plot(steps, costs, color="g", label="$J(\\mathbf{u}^k - stepsize*d^k)$")

        plt.plot(steps, JJ[kk] + descent_arm * steps, color="r", label="$J(\\mathbf{u}^k) - stepsize*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$")

        plt.plot(steps, JJ[kk] + cc * descent_arm * steps, color="g", linestyle="dashed", label="$J(\\mathbf{u}^k) - stepsize*c*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$")

        plt.scatter(stepsizes, costs_armijo, marker="*", color="k")  # plot the tested stepsize

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

    def mpc_plot(xx_star, uu_star, xx_real_mpc, uu_real_mpc, umax, umin, xmax, xmin):
        time = np.arange(constants.TT)

        fig, axs = plt.subplots(constants.NUMBER_OF_STATES + constants.NUMBER_OF_INPUTS, 1, sharex="all")

        for idx in range(constants.NUMBER_OF_STATES):
            axs[idx].plot(time, xx_real_mpc[idx, :], linewidth=2)
            axs[idx].plot(time, xx_star[idx, :], "--r", linewidth=2)
            axs[idx].grid()
            axs[idx].set_ylabel(f"${constants.STATES[idx]}$")

            axs[idx].set_xlim([-1, constants.TT])
            axs[idx].legend(["MPC", "OPT"])

            if xmax < 1.1 * np.amax(xx_real_mpc[idx, :]):
                axs[idx].plot(time, np.ones(constants.TT) * xmax, "--g", linewidth=1)
                axs[idx].plot(time, np.ones(constants.TT) * xmin, "--g", linewidth=1)

        for idx in range(constants.NUMBER_OF_INPUTS):
            axx = idx + constants.NUMBER_OF_STATES

            axs[axx].plot(time, uu_real_mpc[idx, :], "g", linewidth=2)
            axs[axx].plot(time, uu_star[idx, :], "--r", linewidth=2)
            axs[axx].grid()
            axs[axx].set_ylabel(f"${constants.INPUTS[idx]}$")

            if umax < 1.1 * np.amax(uu_real_mpc[idx, :]):
                axs[axx].plot(time, np.ones(constants.TT) * umax, "--g", linewidth=1)
                axs[axx].plot(time, np.ones(constants.TT) * umin, "--g", linewidth=1)

            axs[idx + constants.NUMBER_OF_STATES].set_xlim([-1, constants.TT])
            axs[idx + constants.NUMBER_OF_STATES].legend(["MPC", "OPT"])

        axs[-1].set_xlabel("time")

        fig.align_ylabels(axs)

        plt.show()

    def plot_equilibria(self, eq: Equilibrium):
        # Plot the equilibrium points
        xx = np.array([0, 0, 0, eq.V, eq.beta, eq.psi_dot])
        uu = np.array([eq.delta, eq.Fx])

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
    def dynamics_plot(self, xx_init, uu):
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
