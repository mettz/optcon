import numpy as np
import matplotlib.pyplot as plt
import dynamics

from equilibrium import find_equilibrium_point, nonlinear_system_discretized
import newton_method_optcon as nmo

# import cost functions
import cost as cost

# Allow Ctrl-C to work despite plotting
import signal

signal.signal(signal.SIGINT, signal.SIG_DFL)


if __name__ == "__main__":
    beta_des_1 = 1.0
    V_des_1 = 20.0
    print("Finding initial equilibrium point...")
    initial_eq = find_equilibrium_point(nonlinear_system_discretized, initial_guess=[0, 0, 0], args=(V_des_1, beta_des_1))
    # equilibrium point = [x,y,psi,V,beta,psi_dot]
    # equilibrium input = [delta, Fx]
    steps, trajectory_xx, trajectory_uu = dynamics.trajectory(100, np.array([V_des_1, beta_des_1, initial_eq[0]]), np.array([initial_eq[1], initial_eq[2]]))
    last_index = len(trajectory_xx) - 1
    print(f"Last index: {last_index}")
    print(f"Last state: {trajectory_xx[last_index]}")
    # initial_eq_state = np.array([trajectory_xx[last_index, 0], trajectory_xx[last_index, 1], trajectory_xx[last_index, 2], V_des_1, beta_des_1, initial_eq[0]])
    initial_eq_state = np.array([V_des_1, beta_des_1, initial_eq[0]])
    initial_eq_input = np.array([initial_eq[1], initial_eq[2]])
    print(f"Initial eq point: {initial_eq}") 

    beta_des_2 = 2.0
    V_des_2 = 15.0
    print("Finding final equilibrium point...")
    final_eq = find_equilibrium_point(nonlinear_system_discretized, initial_guess=[0, 0, 0], args=(V_des_2, beta_des_2))
    steps, trajectory_xx, trajectory_uu = dynamics.trajectory(100, np.array([V_des_2, beta_des_2, final_eq[0]]), np.array([final_eq[1], final_eq[2]]))
    print(f"Last index: {last_index}")
    print(f"Last state: {trajectory_xx[last_index]}")
    # final_eq_state = np.array([trajectory_xx[last_index, 0], trajectory_xx[last_index, 1], trajectory_xx[last_index, 2], V_des_2, beta_des_2, final_eq[0]])
    final_eq_state = np.array([V_des_2, beta_des_2, final_eq[0]])
    final_eq_input = np.array([final_eq[1], final_eq[2]])
    print(f"Final eq point:  {final_eq}") 

    xx_ref = np.zeros((nmo.ns, nmo.TT))
    uu_ref = np.zeros((nmo.ni, nmo.TT))
    for i in range(nmo.TT):
        if i < nmo.TT / 2:
            xx_ref[:, i] = initial_eq_state
            uu_ref[:, i] = initial_eq_input
        else:
            xx_ref[:, i] = final_eq_state
            uu_ref[:, i] = final_eq_input

    xx_star, uu_star = nmo.newton_method_optcon(xx_ref, uu_ref)

    tt_hor = np.linspace(0, nmo.tf, nmo.TT)
    fig, axs = plt.subplots(nmo.ns + nmo.ni, 1, sharex="all")

    axs[0].plot(tt_hor, xx_star[0, :], linewidth=2)
    axs[0].plot(tt_hor, xx_ref[0, :], "g--", linewidth=2)
    axs[0].grid()
    axs[0].set_ylabel("$x_1 = V$")
    axs[0].set_xlabel("time")


    axs[1].plot(tt_hor, xx_star[1, :], linewidth=2)
    axs[1].plot(tt_hor, xx_ref[1, :], "g--", linewidth=2)
    axs[1].grid()
    axs[1].set_ylabel("$x_2 = \\beta$")
    axs[1].set_xlabel("time")

    axs[2].plot(tt_hor, xx_star[2, :], linewidth=2)
    axs[2].plot(tt_hor, xx_ref[2, :], "g--", linewidth=2)
    axs[2].grid()
    axs[2].set_ylabel("$x_2 = \\dot{psi}$")
    axs[2].set_xlabel("time")

    axs[3].plot(tt_hor, uu_star[0, :], "r", linewidth=2)
    axs[3].plot(tt_hor, uu_ref[0, :], "r--", linewidth=2)
    axs[3].grid()
    axs[3].set_ylabel("$u_1 = \\delta$")
    axs[3].set_xlabel("time")

    axs[4].plot(tt_hor, uu_star[1, :], "r", linewidth=2)
    axs[4].plot(tt_hor, uu_ref[1, :], "r--", linewidth=2)
    axs[4].grid()
    axs[4].set_ylabel("$u_2 = F_x$")
    axs[4].set_xlabel("time")

    plt.show()


    # #plots.gradient_method_plots(xx_ref, uu_ref, max_iters, xx_star, uu_star, descent, JJ, TT, tf, ni, ns)
    # plots.gradient_method_plots(reference_curve_states, reference_curve_inputs, max_iters, xx_star, uu_star, descent, JJ, TT, tf, ni, ns)

    # #states = ["x", "y", "psi", "V", "beta", "psi_dot"]
    # states = ["V", "beta", "psi_dot"]
    # plt.figure(figsize = (10, 10))
    # plt.clf()
    # plt.title("Reference curve for states")
    # for i in range(np.size(states)):
    #     plt.subplot(3,1,1+i)
    #     plt.plot(reference_curve_states[i, :], label=f"Reference curve {states[i]}")
    #     plt.grid()
    #     plt.legend()
    # #plt.show()

    # inputs = ["delta", "Fx"]
    # plt.figure(figsize = (10, 10))
    # plt.clf()
    # plt.title("Reference curve for inputs")
    # for i in range(np.size(inputs)):
    #     plt.subplot(2,1,1+i)
    #     plt.plot(reference_curve_inputs[i, :], label=f"Reference curve {inputs[i]}")
    #     plt.grid()
    #     plt.legend()
    #plt.show()