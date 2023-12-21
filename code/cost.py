import numpy as np
import dynamics as dyn

number_of_states = dyn.number_of_states  # 3
number_of_inputs = dyn.number_of_inputs  # 2

# QQt = np.array([[10000, 0], [0, 100]]) prof aveva 2 stati e 1 input
# RRt = 1*np.eye(ni)

# nostro caso: QQt 6x6, RRt 2x2
# Sulla diagonale di QQt metto i pesi per gli stati, sulla diagonale di RRt metto i pesi per gli input
# Considerando che x, y e psi sono variabili libere direi che possiamo assegnare loro un peso piccolo, mentre per V, beta e psi_dot un peso molto più grande
# anche perchè l'equilibrio dipende da queste ultime

QQt = np.diag([0.1, 0.1, 0.1])
RRt = np.diag([1, 1])

QQT = QQt


def stagecost(xx, uu, xx_ref, uu_ref):
    """
    Stage-cost

    Quadratic cost function
    l(x,u) = 1/2 (x - x_ref)^T Q (x - x_ref) + 1/2 (u - u_ref)^T R (u - u_ref)

    Args
      - xx \in \R^6 state at time t
      - xx_ref \in \R^6 state reference at time t

      - uu \in \R^2 input at time t
      - uu_ref \in \R^2 input reference at time t


    Return
      - cost at xx,uu
      - gradient of l wrt x, at xx,uu
      - gradient of l wrt u, at xx,uu

    """

    xx = xx[:, None]
    uu = uu[:, None]

    xx_ref = xx_ref[:, None]
    uu_ref = uu_ref[:, None]

    ll = 0.5 * (xx - xx_ref).T @ QQt @ (xx - xx_ref) + 0.5 * (uu - uu_ref).T @ RRt @ (uu - uu_ref)

    lx = QQt @ (xx - xx_ref)
    lu = RRt @ (uu - uu_ref)

    return ll.squeeze(), lx, lu


def termcost(xx, xx_ref):
    """
    Terminal-cost

    Quadratic cost function l_T(x) = 1/2 (x - x_ref)^T Q_T (x - x_ref)

    Args
      - xx \in \R^6 state at time t
      - xx_ref \in \R^6 state reference at time t

    Return
      - cost at xx,uu
      - gradient of l wrt x, at xx,uu
      - gradient of l wrt u, at xx,uu

    """

    xx = xx[:, None]
    xx_ref = xx_ref[:, None]

    llT = 0.5 * (xx - xx_ref).T @ QQT @ (xx - xx_ref)

    lTx = QQT @ (xx - xx_ref)

    return llT.squeeze(), lTx


def hessian_cost():
    """
    Hessian of the stage-cost

    Args
      - xx \in \R^3 state at time t
      - uu \in \R^2 input at time t

    Return
      - hessian of l wrt x, at xx,uu
      - hessian of l wrt u, at xx,uu

    """

    lxx = QQt
    luu = RRt

    lxu = np.zeros((number_of_inputs, number_of_states))

    return lxx, luu, lxu


def hessian_term_cost():
    """
    Hessian of the terminal-cost

    Args
      - xx \in \R^3 state at time t

    Return
      - hessian of l_T wrt x, at xx

    """

    lxxT = QQT

    return lxxT
