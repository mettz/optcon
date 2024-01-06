import cvxpy as cp
import numpy as np

import constants
import dynamics as dyn


def mpc_step(AA, BB, QQ, RR, QQf, xxt, umax=1, umin=-1, x1_max=20, x1_min=-20, x2_max=20, x2_min=-20, T_pred=5):
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


def mpc(xx_star, uu_star):
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
        uu_real_mpc[:, tt] = mpc_step(AA[:, :, tt:], BB[:, :, tt:], QQ, RR, QQT, xx_t_mpc, umax, umin, xmax, xmin, Tpred)

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

    return xx_real_mpc, uu_real_mpc
