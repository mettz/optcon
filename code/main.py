import signal
import argparse
import numpy as np
import matplotlib.pyplot as plt

import constants
import curves
import dynamics as dyn
import equilibrium as eq
import solvers
import mpc
import plots

from LQR_tracking import LQR_tracking

signal.signal(signal.SIGINT, signal.SIG_DFL)


def main(args):
    V_des = [1, 1]
    psi_dot_des = [0, 0.1]

    eq1 = eq.find(V=V_des[0], psi_dot=psi_dot_des[0])
    print(eq1)
    eq2 = eq.find(V=V_des[1], psi_dot=psi_dot_des[1])
    print(eq2)

    xx_eq1 = np.array([V_des[0], eq1.beta, psi_dot_des[0]])
    uu_eq1 = np.array([eq1.delta, eq1.Fx])
    xx_eq2 = np.array([V_des[1], eq2.beta, psi_dot_des[1]])
    uu_eq2 = np.array([eq2.delta, eq2.Fx])

    curve = None
    if args.ref_curve == "step":
        curve = curves.step
    elif args.ref_curve == "cubic":
        curve = curves.cubic_spline
    else:
        raise ValueError(f"Invalid reference curve {curve}")

    xx_ref = curve(start=xx_eq1, end=xx_eq2, steps=constants.TT)
    uu_ref = curve(start=uu_eq1, end=uu_eq2, steps=constants.TT)

    if args.show_ref_curves_plots:
        plt.figure(figsize=(10, 10))
        plt.clf()
        plt.title("Reference curve for states")
        for i in range(constants.NUMBER_OF_STATES):
            plt.subplot(3, 1, 1 + i)
            plt.plot(xx_ref[i, :], label=f"Reference curve {constants.STATES[i]}")
            plt.grid()
            plt.legend()
        plt.show()

        plt.figure(figsize=(10, 10))
        plt.clf()
        plt.title("Reference curve for inputs")
        for i in range(constants.NUMBER_OF_INPUTS):
            plt.subplot(2, 1, 1 + i)
            plt.plot(uu_ref[i, :], label=f"Reference curve {constants.INPUTS[i]}")
            plt.grid()
            plt.legend()
        plt.show()

    if args.show_verify_equilibria:
        plots.verify_equilibria(xx_eq1, uu_eq1, V_des[0], psi_dot_des[0])
        plots.verify_equilibria(xx_eq2, uu_eq2, V_des[1], psi_dot_des[1])

    if args.show_derivative_plots:
        xx_traj = np.ones((constants.NUMBER_OF_STATES, constants.TT))
        uu_traj = np.ones((constants.NUMBER_OF_INPUTS, constants.TT))
        for i in range(constants.TT - 1):
            xx_plus = dyn.dynamics(xx_traj[:, i], uu_traj[:, i])[0]
            xx_traj[:, i + 1] = xx_plus
        plots.derivatives_plot(xx_traj, uu_traj)

    xx_star = None
    uu_star = None

    if args.solver == "gradient":
        xx_star, uu_star = solvers.gradient(xx_ref, uu_ref, show_armijo_plots=args.show_armijo_plots)
    elif args.solver == "newton":
        xx_star, uu_star = solvers.newton(xx_ref, uu_ref, show_armijo_plots=args.show_armijo_plots)
    else:
        raise ValueError(f"Invalid solver {args.solver}")

    tt_hor = np.linspace(0, constants.TF, constants.TT)

    if not args.hide_opt_plots:
        plots.plot_ref_and_star_trajectories(xx_ref, uu_ref, xx_star, uu_star)

    # Defining percentage of errors in state and input
    error = []
    for i in range(constants.NUMBER_OF_STATES):
        error.append(np.abs(xx_ref[i, :] - xx_star[i, :]))
        print(f"Error in state {constants.STATES[i]}: {np.mean(error)}")

    for i in range(constants.NUMBER_OF_INPUTS):
        error.append(np.abs(uu_ref[i, :] - uu_star[i, :]))
        print(f"Error in input {constants.INPUTS[i]}: {np.mean(error)}")

    # Defining overshooting in input
    overshooting = []
    for i in range(constants.NUMBER_OF_INPUTS):
        max_input_star = np.max(uu_star[i, :])
        max_input_ref = np.max(uu_ref[i, :])
        overshooting.append((max_input_star - max_input_ref) / max_input_ref)
        print(f"Overshooting in input {constants.INPUTS[i]}: {overshooting[i]}")

    if args.lqr:
        xx_tracking, uu_tracking = LQR_tracking(xx_star, uu_star)

        plt.title("Trajectory tracking via LQR")
        for i in range(constants.NUMBER_OF_STATES):
            plt.subplot(constants.NUMBER_OF_STATES, 1, 1 + i)
            plt.plot(tt_hor, xx_star[i, :], label=f"Reference curve {constants.STATES[i]}")
            plt.plot(tt_hor, xx_tracking[i, :], label=f"State {constants.STATES[i]}")

            plt.grid()
            plt.legend()
        plt.show()

        plt.figure()
        plt.clf()
        plt.title("Trajectory tracking inputs")
        for i in range(constants.NUMBER_OF_INPUTS):
            plt.subplot(constants.NUMBER_OF_INPUTS, 1, 1 + i)
            plt.plot(tt_hor, uu_star[i, :], label=f"Reference curve {constants.INPUTS[i]}")
            plt.plot(tt_hor, uu_tracking[i, :], label=f"State {constants.INPUTS[i]}")

            plt.grid()
            plt.legend()
        plt.show()

    if args.mpc:
        Tpred = 50
        umax = 1
        umin = -umax
        xmax = 20
        xmin = -xmax

        xx0 = xx_star[:, 0] - np.array([0.001, 0.001, 0.001])
        AA = np.zeros((constants.NUMBER_OF_STATES, constants.NUMBER_OF_STATES, constants.TT))
        BB = np.zeros((constants.NUMBER_OF_STATES, constants.NUMBER_OF_INPUTS, constants.TT))

        QQ = np.diag([1, 400, 400])
        QQT = QQ * 700
        RR = np.diag([0.5, 0.5])

        for tt in range(constants.TT):
            fx, fu = dyn.dynamics(xx_star[:, tt], uu_star[:, tt])[1:]

            AA[:, :, tt] = fx.T
            BB[:, :, tt] = fu.T

        xx_real_mpc = np.ones((constants.NUMBER_OF_STATES, constants.TT))
        uu_real_mpc = np.zeros((constants.NUMBER_OF_INPUTS, constants.TT))

        xx_real_mpc[:, 0] = xx0.squeeze()

        for tt in range(constants.TT - 1):
            # System evolution - real with MPC
            xx_t_mpc = xx_real_mpc[:, tt]  # get initial condition

            # Solve the MPC problem
            uu_real_mpc[:, tt] = mpc.linear_mpc(AA[:, :, tt:], BB[:, :, tt:], QQ, RR, QQT, xx_t_mpc, umax, umin, xmax, xmin, Tpred)

            xx_real_mpc[:, tt + 1] = dyn.dynamics(xx_real_mpc[:, tt], uu_real_mpc[:, tt])[0]

        # For plotting purposes
        xx_real_mpc[:, -1] = xx_real_mpc[:, -Tpred - 1]
        xx_real_mpc[:, -2] = xx_real_mpc[:, -Tpred - 1]
        xx_real_mpc[:, -3] = xx_real_mpc[:, -Tpred - 1]
        xx_real_mpc[:, -4] = xx_real_mpc[:, -Tpred - 1]

        uu_real_mpc[:, -1] = uu_real_mpc[:, -Tpred - 1]
        uu_real_mpc[:, -2] = uu_real_mpc[:, -Tpred - 1]
        uu_real_mpc[:, -3] = uu_real_mpc[:, -Tpred - 1]
        uu_real_mpc[:, -4] = uu_real_mpc[:, -Tpred - 1]
        plots.mpc_plot(xx_ref, uu_ref, xx_real_mpc, uu_real_mpc, umax, umin, xmax, xmin)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous Car Optimization")

    parser.add_argument("-c", "--ref-curve", type=str, choices=["step", "cubic"], default="step", help="Reference curve to follow")

    parser.add_argument("-s", "--solver", type=str, choices=["gradient", "newton"], default="newton", help="Solver to use")

    parser.add_argument("--lqr", action="store_true", default=False, help="Use LQR")

    parser.add_argument("--mpc", action="store_true", default=False, help="Use MPC")

    parser.add_argument("--show-ref-curves-plots", action="store_true", default=False, help="Show the plots of the reference curve")

    parser.add_argument("--show-verify-equilibria", action="store_true", default=False, help="Show the plots of the verify equilibria")

    parser.add_argument("--show-derivative-plots", action="store_true", default=False, help="Show the plots of the derivatives")

    parser.add_argument("--show-armijo-plots", action="store_true", default=False, help="Show the Armijo plots")

    parser.add_argument("--hide-opt-plots", action="store_true", default=False, help="Hide the plots of the optimization")

    main(parser.parse_args())
