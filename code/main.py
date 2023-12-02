import numpy as np
import matplotlib.pyplot as plt
import dynamics
import plots

from equilibrium import find_equilibrium_point, nonlinear_system_discretized
from gradient_method_optcon import gradient_method, max_iters, descent, JJ, TT, tf, ni, ns

# import cost functions
import cost as cost

# Allow Ctrl-C to work despite plotting
import signal

signal.signal(signal.SIGINT, signal.SIG_DFL)


if __name__ == "__main__":
    beta_des_1 = 1.0
    V_des_1 = 20.0
    initial_eq = find_equilibrium_point(nonlinear_system_discretized, initial_guess=[0, 0, 0], args=(V_des_1, beta_des_1))
    # equilibrium point = [x,y,psi,V,beta,psi_dot]
    # equilibrium input = [delta, Fx]
    steps, trajectory_xx, trajectory_uu = dynamics.trajectory(100, np.array([0, 0, 0, V_des_1, beta_des_1, initial_eq[0]]), np.array([initial_eq[1], initial_eq[2]]))
    last_index = len(trajectory_xx) - 1
    print(f"Last index: {last_index}")
    print(f"Last state: {trajectory_xx[last_index]}")
    initial_eq_state = np.array([trajectory_xx[last_index, 0], trajectory_xx[last_index, 1], trajectory_xx[last_index, 2], V_des_1, beta_des_1, initial_eq[0]])
    initial_eq_input = np.array([initial_eq[1], initial_eq[2]])
    print(f"Initial eq point: {initial_eq}")

    beta_des_2 = 2.0
    V_des_2 = 15.0
    final_eq = find_equilibrium_point(nonlinear_system_discretized, initial_guess=[0, 0, 0], args=(V_des_2, beta_des_2))
    steps, trajectory_xx, trajectory_uu = dynamics.trajectory(100, np.array([0, 0, 0, V_des_2, beta_des_2, final_eq[0]]), np.array([final_eq[1], final_eq[2]]))
    print(f"Last index: {last_index}")
    print(f"Last state: {trajectory_xx[last_index]}")
    final_eq_state = np.array([trajectory_xx[last_index, 0], trajectory_xx[last_index, 1], trajectory_xx[last_index, 2], V_des_2, beta_des_2, final_eq[0]])
    final_eq_input = np.array([final_eq[1], final_eq[2]])

    print(f"Final eq poin:  {final_eq}")

    # Building transition between equilibria
    time = 10000
    reference_curve_states = np.zeros((6, time))
    reference_curve_inputs = np.zeros((2, time))
    for i in range(time):
        if i < time / 2:
            reference_curve_states[:, i] = initial_eq_state
            reference_curve_inputs[:, i] = initial_eq_input
        else:
            reference_curve_states[:, i] = final_eq_state
            reference_curve_inputs[:, i] = final_eq_input

    xx_star, uu_star = gradient_method(reference_curve_states, reference_curve_inputs)

    # gradient_method_plots(xx_ref, uu_ref, max_iters, xx_star, uu_star, descent, JJ, TT, tf, ni, ns)
    plots.gradient_method_plots(reference_curve_states, reference_curve_inputs, max_iters, xx_star, uu_star, descent, JJ, TT, tf, ni, ns)

    states = ["x", "y", "psi", "V", "beta", "psi_dot"]
    plt.figure()
    plt.clf()
    plt.title("Reference curve for states")
    for i in range(np.size(reference_curve_states, 1)):
        plt.plot(reference_curve_states[i, :], label=f"Reference curve {states[i]}")
        plt.legend()
        plt.grid()
        # plt.show()

    inputs = ["delta", "Fx"]
    plt.figure()
    plt.clf()
    plt.title("Reference curve for inputs")
    for i in range(np.size(reference_curve_inputs, 1)):
        plt.plot(reference_curve_inputs[i, :], label=f"Reference curve {inputs[i]}")
        plt.legend()
        plt.grid()
        # plt.show()


"""
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

    return x"""
