import signal
import argparse
import numpy as np
import matplotlib.pyplot as plt

import constants
import curves
import dynamics as dyn
import equilibrium as eq
import gradient_method_optcon as gmo
import newton_method_optcon as nmo
import plots

signal.signal(signal.SIGINT, signal.SIG_DFL)


def main(args):
    V_des = [20, 20]
    beta_des = [1, 2]

    eq1 = eq.find(V=V_des[0], beta=beta_des[0])
    print(eq1)
    eq2 = eq.find(V=V_des[1], beta=beta_des[1])
    print(eq2)

    xx_eq1 = np.array([V_des[0], beta_des[0], eq1.psi_dot])
    uu_eq1 = np.array([eq1.delta, eq1.Fx])
    xx_eq2 = np.array([V_des[1], beta_des[1], eq2.psi_dot])
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
        plots.verify_equilibria(xx_eq1, uu_eq1, V_des[0], beta_des[0])
        plots.verify_equilibria(xx_eq2, uu_eq2, V_des[1], beta_des[1])

    if args.show_derivative_plots:
        xx_traj = np.ones((constants.NUMBER_OF_STATES, constants.TT))
        uu_traj = np.ones((constants.NUMBER_OF_INPUTS, constants.TT))
        for i in range(constants.TT-1): 
            xx_plus = dyn.dynamics(xx_traj[:,i], uu_traj[:,i])[0]
            xx_traj[:, i+1] = xx_plus
        plots.derivatives_plot(xx_traj, uu_traj)

    # Application of the newthon method
    #xx_star, uu_star = nmo.newton_method_optcon(xx_ref, uu_ref)
    xx_star, uu_star = gmo.gradient_method(xx_ref, uu_ref)
    print("xx_star", xx_star.shape)
    print("uu_star", uu_star.shape)
    print("xx_ref", xx_ref.shape)
    print("uu_ref", uu_ref.shape)

    tt_hor = np.linspace(0, constants.TF, constants.TT)
    plt.figure()
    plt.clf()
    plt.title("Trajectory following")
    for i in range(constants.NUMBER_OF_STATES):
        plt.subplot(constants.NUMBER_OF_STATES, 1, 1 + i)
        plt.plot(tt_hor, xx_ref[i, :], label=f"Reference curve {constants.STATES[i]}")
        plt.plot(tt_hor, xx_star[i, :], label=f"State {constants.STATES[i]}")

        plt.grid()
        plt.legend()
    plt.show()

    tt_hor = np.linspace(0, constants.TF, constants.TT)
    plt.figure()
    plt.clf()
    plt.title("Trajectory following inputs")
    for i in range(constants.NUMBER_OF_INPUTS):
        plt.subplot(constants.NUMBER_OF_INPUTS, 1, 1 + i)
        plt.plot(tt_hor, uu_ref[i, :], label=f"Reference curve {constants.INPUTS[i]}")
        plt.plot(tt_hor, uu_star[i, :], label=f"State {constants.INPUTS[i]}")

        plt.grid()
        plt.legend()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous Car Optimization")

    parser.add_argument("-c", "--ref-curve", type=str, choices=["step", "cubic"], default="step", help="Reference curve to follow")

    parser.add_argument("--show-ref-curves-plots", action="store_true", default=False, help="Show the plots of the reference curve")

    parser.add_argument("--show-verify-equilibria", action="store_true", default=False, help="Show the plots of the verify equilibria")

    parser.add_argument("--show-derivative-plots", action="store_true", default=False, help="Show the plots of the derivatives")

    main(parser.parse_args())

    # # Application of the newthon method
    # # xx_star, uu_star = nmo.newton_method_optcon(xx_ref, uu_ref)
    # xx_star, uu_star = gradient_method(xx_ref, uu_ref)
    # # print("xx_star", xx_star.shape)
    # # print("uu_star", uu_star.shape)
    # # print("xx_ref", xx_ref.shape)
    # # print("uu_ref", uu_ref.shape)

    

    # # fig, axs = plt.subplots(nmo.ns + nmo.ni, 1, sharex="all")

    # # axs[0].plot(tt_hor, xx_star[0, :], linewidth = 2)
    # # axs[0].plot(tt_hor, xx_ref[0, :], "g--", linewidth = 2)
    # # axs[0].grid()
    # # axs[0].set_ylabel("$x_4 = V$")
    # # axs[0].set_xlabel("time")

    # # axs[1].plot(tt_hor, xx_star[1, :], linewidth = 2)
    # # axs[1].plot(tt_hor, xx_ref[1, :], "g--", linewidth = 2)
    # # axs[1].grid()
    # # axs[1].set_ylabel("$x_5 = \\beta$")
    # # axs[1].set_xlabel("time")

    # # axs[2].plot(tt_hor, xx_star[2, :], linewidth = 2)
    # # axs[2].plot(tt_hor, xx_ref[2, :], "g--", linewidth = 2)
    # # axs[2].grid()
    # # axs[2].set_ylabel("$x_6 = \\dot{psi}$")
    # # axs[2].set_xlabel("time")

    # # axs[3].plot(tt_hor, uu_star[0, :], "r", linewidth = 2)
    # # axs[3].plot(tt_hor, uu_ref[0, :], "r--", linewidth = 2)
    # # axs[3].grid()
    # # axs[3].set_ylabel("$u_1 = \\delta$")
    # # axs[3].set_xlabel("time")

    # # axs[4].plot(tt_hor, uu_star[1, :], "r", linewidth = 2)
    # # axs[4].plot(tt_hor, uu_ref[1, :], "r--", linewidth = 2)
    # # axs[4].grid()
    # # axs[4].set_ylabel("$u_2 = F_x$")
    # # axs[4].set_xlabel("time")

    # # plt.show()
