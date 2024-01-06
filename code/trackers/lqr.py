import numpy as np
import dynamics as dyn
import matplotlib.pyplot as plt

import constants


def lqr(xx_star, uu_star):
    x0 = xx_star[:, 0] - np.array([0.1, 0.1, 0.1])

    # Step 0: define the cost matrices
    QQ_reg = np.diag([0.1, 1, 10])
    QQ_reg_T = QQ_reg
    RR_reg = np.diag([15, 0.1])

    # Step 1: linearize the dynamics around the optimal trajectory (slide 3 pacco 9)
    AA = np.zeros((constants.NUMBER_OF_STATES, constants.NUMBER_OF_STATES, constants.TT - 1))
    BB = np.zeros((constants.NUMBER_OF_STATES, constants.NUMBER_OF_INPUTS, constants.TT - 1))
    for tt in range(constants.TT - 1):
        AA[:, :, tt] = dyn.dynamics(xx_star[:, tt], uu_star[:, tt])[1].T
        BB[:, :, tt] = dyn.dynamics(xx_star[:, tt], uu_star[:, tt])[2].T

    PP = np.zeros((constants.NUMBER_OF_STATES, constants.NUMBER_OF_STATES, constants.TT))
    KK = np.zeros((constants.NUMBER_OF_INPUTS, constants.NUMBER_OF_STATES, constants.TT))

    PP[:, :, -1] = QQ_reg_T

    # Solve Riccati equation
    for tt in reversed(range(constants.TT - 1)):
        QQt = QQ_reg
        RRt = RR_reg
        AAt = AA[:, :, tt]
        BBt = BB[:, :, tt]
        PPtp = PP[:, :, tt + 1]

        PP[:, :, tt] = QQt + AAt.T @ PPtp @ AAt - +(AAt.T @ PPtp @ BBt) @ np.linalg.inv((RRt + BBt.T @ PPtp @ BBt)) @ (BBt.T @ PPtp @ AAt)

    # Evaluate KK
    for tt in range(constants.TT - 1):
        QQt = QQ_reg
        RRt = RR_reg
        AAt = AA[:, :, tt]
        BBt = BB[:, :, tt]
        PPtp = PP[:, :, tt + 1]

        KK[:, :, tt] = -np.linalg.inv(RRt + BBt.T @ PPtp @ BBt) @ (BBt.T @ PPtp @ AAt)

    xx = np.zeros((constants.NUMBER_OF_STATES, constants.TT))
    uu = np.zeros((constants.NUMBER_OF_INPUTS, constants.TT))

    xx[:, 0] = x0

    for tt in range(constants.TT - 1):
        uu[:, tt] = uu_star[:, tt] + KK[:, :, tt] @ (xx[:, tt] - xx_star[:, tt])
        xx[:, tt + 1] = dyn.dynamics(xx[:, tt], uu[:, tt])[0]

    # Calculate tracking error
    tracking_error_uu = uu - uu_star
    tracking_error_xx = xx - xx_star

    # Plot tracking error
    plt.figure()
    plt.title("Tracking error")
    plt.subplot(2, 1, 1)
    plt.grid()
    plt.plot(tracking_error_uu.T)
    plt.xlabel("Time")
    plt.ylabel("uu-uu_star")
    plt.subplot(2, 1, 2)
    plt.plot(tracking_error_xx.T)
    plt.xlabel("Time")
    plt.ylabel("xx-xx_star")
    plt.show()

    return xx, uu
