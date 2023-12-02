import numpy as np
import matplotlib.pyplot as plt
import dynamics
from plots import plot_equilibria
from plots import plot_with_equilibria

from equilibrium import find_equilibrium_point, nonlinear_system_discretized

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
    steps, trajectory_xx, trajectory_uu = dynamics.trajectory(100, np.array([0, 0, 0, V_des_1, beta_des_1, final_eq[0]]), np.array([final_eq[1], final_eq[2]]))
    print(f"Last index: {last_index}")
    print(f"Last state: {trajectory_xx[last_index]}")
    final_eq_state = np.array([trajectory_xx[last_index, 0], trajectory_xx[last_index, 1], trajectory_xx[last_index, 2], V_des_2, beta_des_2, initial_eq[0]])
    final_eq_input = np.array([final_eq[1], final_eq[2]])

    print(f"Final eq poin:  {final_eq}")



    #Building transition between equilibria
    time = 10000
    time_points = np.arange(time)
    ref_curve2 = np.vstack((time_points, np.where(time_points < time/2, initial_eq, final_eq))).T
    # reference_curve = np.zeros((time, len(initial_eq)))
    # for i in range(time):
    #     if i < time/2:
    #         reference_curve[i] = np.array([i, initial_eq])
    #     else:
    #         reference_curve[i] = np.array([i, final_eq])

    #ref_curve2 = np.ones(int(time / 2)) * initial_eq + np.ones(int(time / 2)) * final_eq
    
    plt.figure()
    plt.clf()
    #plt.plot(ref_curve2)
    plt.title("Reference curve")
    plt.grid()
    plt.legend()
    plt.show()

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