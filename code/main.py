import signal
import argparse
import numpy as np
import matplotlib.pyplot as plt

import constants
import curves
import dynamics
import equilibrium as eq

signal.signal(signal.SIGINT, signal.SIG_DFL)


def main(args):
    V_des = [20, 15]
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous Car Optimization")

    parser.add_argument("-c", "--ref-curve", type=str, choices=["step", "cubic"], default="step", help="Reference curve to follow")

    parser.add_argument("--show-ref-curves-plots", action="store_true", default=False, help="Show the plots of the reference curve")

    main(parser.parse_args())

    # Definition of flags that enable or disable the plots
    # see_reference_curve_plots = False

    # V_des_1 = 20.0
    # beta_des_1 = 1.0
    # print("Finding initial equilibrium point...")
    # initial_eq = find_equilibrium_point(nonlinear_system_discretized, initial_guess=[0, 0, 0], args=(V_des_1, beta_des_1))
    # # initial_eq contiene psi_dot, delta e Fx del primo punto di equiilibrio
    # # equilibrium point = [x,y,psi,V,beta,psi_dot]
    # # equilibrium input = [delta, Fx]
    # steps, trajectory_xx, trajectory_uu = dynamics.trajectory(100, np.array([V_des_1, beta_des_1, initial_eq[0]]), np.array([initial_eq[1], initial_eq[2]]))
    # last_index = len(trajectory_xx) - 1
    # initial_eq_state = np.array([V_des_1, beta_des_1, initial_eq[0]])
    # initial_eq_input = np.array([initial_eq[1], initial_eq[2]])
    # print(f"Initial eq state: {initial_eq_state}")
    # print(f"Initial eq input: {initial_eq_input}")

    # points = 2500
    # x = np.ones((3, points))

    # x_taylor = np.ones((3, points))
    # t_taylor = 1850
    # f_a = np.ones((3, 1))
    # f_a[:, 0] = dynamics.dynamics(x[:, t_taylor], initial_eq_input)[1][0]

    # print(np.shape(f_a))
    # for tt in range(points - 1):
    #     x[:, tt + 1] = dynamics.dynamics(x[:, tt], initial_eq_input)[0]

    #     diff = x[:, tt] - x[:, t_taylor]

    #     x_taylor[:, tt + 1][0] = dynamics.dynamics(x[:, t_taylor], initial_eq_input)[0][0] + f_a.T @ diff

    # # Plotting the entire vector x
    # plt.figure()
    # plt.plot(x[0, :], label="V")
    # plt.plot(x_taylor[0, :], label="V Taylor")
    # plt.xlabel("Time")
    # plt.ylabel("State")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # V_des_2 = 15.0
    # beta_des_2 = 2.0
    # print("Finding final equilibrium point...")
    # final_eq = find_equilibrium_point(nonlinear_system_discretized, initial_guess=[0, 0, 0], args=(V_des_2, beta_des_2))
    # x_init_2 = 0.0
    # y_init_2 = 0.0
    # psi_init_2 = 0.0
    # steps, trajectory_xx, trajectory_uu = dynamics.trajectory(100, np.array([V_des_2, beta_des_2, final_eq[0]]), np.array([final_eq[1], final_eq[2]]))
    # final_eq_state = np.array([V_des_2, beta_des_2, final_eq[0]])
    # final_eq_input = np.array([final_eq[1], final_eq[2]])
    # print(f"Final eq state: {final_eq_state}")
    # print(f"Final eq input: {final_eq_input}")

    # """Ho importato queste due righe di codice dall'altro branch, probabilmente vanno verificate le dimensioni"""
    # # Verification of the equilibrium points
    # # plots.verify_equilibria(initial_eq_state, initial_eq_input, V_des_1, beta_des_1)
    # # plots.verify_equilibria(final_eq_state, final_eq_input, V_des_2, beta_des_2)

    # # Step reference curve
    # xx_ref, uu_ref = step(initial_eq_state, initial_eq_input, final_eq_state, final_eq_input)
    # # Smooth reference curve
    # # xx_ref, uu_ref = smooth_trajectory(initial_eq_state, initial_eq_input, final_eq_state, final_eq_input)

    # # Plot of the reference curve
    # """
    # The plots of the refernce curve are printed if the variable see_reference_curve_plots is set to True
    # """
    # states = ["V", "beta", "psi_dot"]
    # inputs = ["delta", "Fx"]

    # if see_reference_curve_plots:
    #     plt.figure(figsize=(10, 10))
    #     plt.clf()
    #     plt.title("Reference curve for states")
    #     for i in range(np.size(states)):
    #         plt.subplot(3, 2, 1 + i)
    #         plt.plot(xx_ref[i, :], label=f"Reference curve {states[i]}")
    #         plt.grid()
    #         plt.legend()
    #     plt.show()

    #     plt.figure(figsize=(10, 10))
    #     plt.clf()
    #     plt.title("Reference curve for inputs")
    #     for i in range(np.size(inputs)):
    #         plt.subplot(2, 1, 1 + i)
    #         plt.plot(uu_ref[i, :], label=f"Reference curve {inputs[i]}")
    #         plt.grid()
    #         plt.legend()
    #     plt.show()

    # # Application of the newthon method
    # # xx_star, uu_star = nmo.newton_method_optcon(xx_ref, uu_ref)
    # xx_star, uu_star = gradient_method(xx_ref, uu_ref)
    # # print("xx_star", xx_star.shape)
    # # print("uu_star", uu_star.shape)
    # # print("xx_ref", xx_ref.shape)
    # # print("uu_ref", uu_ref.shape)

    # tt_hor = np.linspace(0, nmo.tf, nmo.TT)
    # plt.figure()
    # plt.clf()
    # plt.title("Trajectory following")
    # for i in range(np.size(states)):
    #     plt.subplot(3, 1, 1 + i)
    #     plt.plot(tt_hor, xx_ref[i, :], label=f"Reference curve {states[i]}")
    #     plt.plot(tt_hor, xx_star[i, :], label=f"State {states[i]}")

    #     plt.grid()
    #     plt.legend()
    # plt.show()

    # tt_hor = np.linspace(0, nmo.tf, nmo.TT)
    # plt.figure()
    # plt.clf()
    # plt.title("Trajectory following inputs")
    # for i in range(np.size(inputs)):
    #     plt.subplot(2, 1, 1 + i)
    #     plt.plot(tt_hor, uu_ref[i, :], label=f"Reference curve {inputs[i]}")
    #     plt.plot(tt_hor, uu_star[i, :], label=f"State {inputs[i]}")

    #     plt.grid()
    #     plt.legend()
    # plt.show()

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
