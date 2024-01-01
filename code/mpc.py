import cvxpy as cp

import constants


def linear_mpc(AA, BB, QQ, RR, QQf, xxt, umax=1, umin=-1, x1_max=20, x1_min=-20, x2_max=20, x2_min=-20, T_pred=5):
    """
    Linear MPC solver - Constrained LQR

    Given a measured state xxt measured at t
    gives back the optimal input to be applied at t

    Args
      - AA, BB: linear dynamics
      - QQ,RR,QQf: cost matrices
      - xxt: initial condition (at time t)
      - T: time (prediction) horizon

    Returns
      - u_t: input to be applied at t
      - xx, uu predicted trajectory

    """

    xxt = xxt.squeeze()
    samples = AA.shape[2]

    xx_mpc = cp.Variable((constants.NUMBER_OF_STATES, T_pred))
    uu_mpc = cp.Variable((constants.NUMBER_OF_INPUTS, T_pred))

    cost = 0
    constr = []

    for tau in range(min(T_pred - 1, samples)):
        AAt = AA[:, :, tau]
        BBt = BB[:, :, tau]
        cost += cp.quad_form(xx_mpc[:, tau], QQ) + cp.quad_form(uu_mpc[:, tau], RR)
        constr += [
            xx_mpc[:, tau + 1] == AAt @ xx_mpc[:, tau] + BBt @ uu_mpc[:, tau],  # dynamics constraint
        ]
    # sums problem objectives and concatenates constraints.
    cost += cp.quad_form(xx_mpc[:, T_pred - 1], QQf)
    constr += [xx_mpc[:, 0] == xxt]

    problem = cp.Problem(cp.Minimize(cost), constr)
    problem.solve()

    if problem.status == "infeasible":
        # Otherwise, problem.value is inf or -inf, respectively.
        print("Infeasible problem! CHECK YOUR CONSTRAINTS!!!")

    return uu_mpc[:, 0].value
