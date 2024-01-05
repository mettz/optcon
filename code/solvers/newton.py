import matplotlib.pyplot as plt
import numpy as np

import constants
import cost as cst
import dynamics as dyn
import plots

plt.rcParams["figure.figsize"] = (10, 8)
plt.rcParams.update({"font.size": 22})

# Algorithm parameters
max_iters = 40
stepsize_0 = 5.0

# Armijo parametrs
cc = 0.5
beta = 0.7
armijo_maxiters = 20
term_cond = 1e-3
visu_armijo = True

# Initial guess
xx_init = np.ones((constants.NUMBER_OF_STATES, constants.TT))
uu_init = np.ones((constants.NUMBER_OF_INPUTS, constants.TT))

# State and input sequences
xx = np.zeros((constants.NUMBER_OF_STATES, constants.TT, max_iters))
uu = np.zeros((constants.NUMBER_OF_INPUTS, constants.TT, max_iters))

# Lambda - costate sequence
lmbd = np.zeros((constants.NUMBER_OF_STATES, constants.TT, max_iters))

# Du - descent direction
deltau = np.zeros((constants.NUMBER_OF_INPUTS, constants.TT, max_iters))
deltax = np.zeros((constants.NUMBER_OF_STATES, constants.TT, max_iters))

# DJ - gradient of J wrt u
dJ = np.zeros((constants.NUMBER_OF_INPUTS, constants.TT, max_iters))

# Cost and descent
JJ = np.zeros(max_iters)
descent = np.zeros(max_iters)
descent_arm = np.zeros(max_iters)

# Affine terms
qq = np.zeros((constants.NUMBER_OF_STATES, constants.TT))
rr = np.zeros((constants.NUMBER_OF_INPUTS, constants.TT))

# Riccati terms
AA = np.zeros((constants.NUMBER_OF_STATES, constants.NUMBER_OF_STATES, constants.TT))
BB = np.zeros((constants.NUMBER_OF_STATES, constants.NUMBER_OF_INPUTS, constants.TT))
PP = np.zeros((constants.NUMBER_OF_STATES, constants.NUMBER_OF_STATES, constants.TT))
pp = np.zeros((constants.NUMBER_OF_STATES, constants.TT))

# KK and sigma
KK = np.zeros((constants.NUMBER_OF_INPUTS, constants.NUMBER_OF_STATES, constants.TT))
sigma = np.zeros((constants.NUMBER_OF_INPUTS, constants.TT))


def newton_method(xx_ref, uu_ref, **kwargs):
    global max_iters
    visu_armijo = kwargs.get("visu_armijo", False)

    print("Starting the computation of the optimal trajectory...")

    fxx, fuu, fxu = cst.hessian_cost()
    QQ = fxx[:, :, None].repeat(constants.TT, axis=2)
    RR = fuu[:, :, None].repeat(constants.TT, axis=2)
    SS = fxu[:, :, None].repeat(constants.TT, axis=2)

    # Initialization of the trajectory to the initial equilibrium point
    xx[:, :, 0] = xx_ref[:, 0, None]
    uu[:, :, 0] = uu_ref[:, 0, None]

    x0 = xx_ref[:, 0]

    for kk in range(max_iters - 1):
        JJ[kk] = 0  # Cost initialization

        # Computation of the cost
        for tt in range(constants.TT - 1):
            temp_cost = cst.stagecost(xx[:, tt, kk], uu[:, tt, kk], xx_ref[:, tt], uu_ref[:, tt])[0]
            JJ[kk] += temp_cost

        temp_cost = cst.termcost(xx[:, -1, kk], xx_ref[:, -1])[0]
        JJ[kk] += temp_cost

        lmbd_temp = cst.termcost(xx[:, -1, kk], xx_ref[:, -1])[1]
        lmbd[:, -1, kk] = lmbd_temp.squeeze()
        QQ[:, :, -1] = cst.hessian_term_cost()
        qq[:, -1] = cst.stagecost(xx[:, -1, kk], uu[:, -1, kk], xx_ref[:, -1], uu_ref[:, -1])[1].squeeze()

        for tt in reversed(range(constants.TT - 1)):  # integration backward in time
            at, bt = cst.stagecost(xx[:, tt, kk], uu[:, tt, kk], xx_ref[:, tt], uu_ref[:, tt])[1:]
            fx, fu = dyn.dynamics(xx[:, tt, kk], uu[:, tt, kk])[1:]

            At = fx.T
            Bt = fu.T

            # Fill the affine terms
            qq[:, tt] = at.squeeze()
            rr[:, tt] = bt.squeeze()

            # Fill the Riccati terms
            AA[:, :, tt] = At
            BB[:, :, tt] = Bt

            lmbd_temp = At.T @ lmbd[:, tt + 1, kk][:, None] + at  # costate equation
            dJ_temp = Bt.T @ lmbd[:, tt + 1, kk][:, None] + bt  # gradient of J wrt u

            lmbd[:, tt, kk] = lmbd_temp.squeeze()
            dJ[:, tt, kk] = dJ_temp.squeeze()

        # Solve Riccati equation
        PP[:, :, -1] = QQ[:, :, -1]
        pp[:, -1] = qq[:, -1]
        for tt in reversed(range(constants.TT - 1)):
            QQt = QQ[:, :, tt]
            qqt = qq[:, tt][:, None]
            RRt = RR[:, :, tt]
            rrt = rr[:, tt][:, None]
            AAt = AA[:, :, tt]
            BBt = BB[:, :, tt]
            SSt = SS[:, :, tt]
            PPtp = PP[:, :, tt + 1]
            pptp = pp[:, tt + 1][:, None]

            MMt_inv = np.linalg.inv(RRt + BBt.T @ PPtp @ BBt)
            mmt = rrt + BBt.T @ pptp

            PPt = AAt.T @ PPtp @ AAt - (BBt.T @ PPtp @ AAt + SSt).T @ MMt_inv @ (BBt.T @ PPtp @ AAt + SSt) + QQt
            ppt = AAt.T @ pptp - (BBt.T @ PPtp @ AAt + SSt).T @ MMt_inv @ mmt + qqt

            PP[:, :, tt] = PPt
            pp[:, tt] = ppt.squeeze()

        # Evaluate KK and sigma
        for tt in range(constants.TT - 1):
            QQt = QQ[:, :, tt]
            qqt = qq[:, tt][:, None]
            RRt = RR[:, :, tt]
            rrt = rr[:, tt][:, None]
            AAt = AA[:, :, tt]
            BBt = BB[:, :, tt]
            SSt = SS[:, :, tt]

            PPtp = PP[:, :, tt + 1]
            pptp = pp[:, tt + 1][:, None]

            # Check positive definiteness
            MMt_inv = np.linalg.inv(RRt + BBt.T @ PPtp @ BBt)
            mmt = rrt + BBt.T @ pptp

            KK[:, :, tt] = -MMt_inv @ (BBt.T @ PPtp @ AAt + SSt)
            sigma_t = -MMt_inv @ mmt
            sigma[:, tt] = sigma_t.squeeze()

        # Compute the descent direction
        for tt in range(constants.TT - 1):
            deltau[:, tt, kk] = KK[:, :, tt] @ deltax[:, tt, kk] + sigma[:, tt]
            deltax[:, tt + 1, kk] = AA[:, :, tt] @ deltax[:, tt, kk] + BB[:, :, tt] @ deltau[:, tt, kk]
            descent[kk] += deltau[:, tt, kk].T @ deltau[:, tt, kk]
            descent_arm[kk] += dJ[:, tt, kk].T @ deltau[:, tt, kk]

        # ##################################
        # # Stepsize selection - ARMIJO
        # ##################################

        stepsizes = []  # list of stepsizes
        costs_armijo = []

        stepsize = stepsize_0

        for ii in range(armijo_maxiters):
            # temp solution update

            xx_temp = np.zeros((constants.NUMBER_OF_STATES, constants.TT))
            uu_temp = np.zeros((constants.NUMBER_OF_INPUTS, constants.TT))

            xx_temp[:, 0] = x0

            for tt in range(constants.TT - 1):
                uu_temp[:, tt] = uu[:, tt, kk] + stepsize * deltau[:, tt, kk]
                xx_temp[:, tt + 1] = dyn.dynamics(xx_temp[:, tt], uu_temp[:, tt])[0]

            # temp cost calculation
            JJ_temp = 0

            for tt in range(constants.TT - 1):
                temp_cost = cst.stagecost(xx_temp[:, tt], uu_temp[:, tt], xx_ref[:, tt], uu_ref[:, tt])[0]
                JJ_temp += temp_cost

            temp_cost = cst.termcost(xx_temp[:, -1], xx_ref[:, -1])[0]
            JJ_temp += temp_cost

            stepsizes.append(stepsize)  # save the stepsize
            costs_armijo.append(np.min([JJ_temp, 100 * JJ[kk]]))  # save the cost associated to the stepsize

            if JJ_temp > JJ[kk] + cc * stepsize * descent_arm[kk]:
                # update the stepsize
                stepsize = beta * stepsize
            else:
                print("Armijo stepsize = {:.3e}".format(stepsize))
                break

        if visu_armijo and kk % 10 == 0:
            plots.armijo_plot(stepsize_0, stepsizes, costs_armijo, descent_arm[kk], JJ, kk, cc, x0, uu, deltau, dyn, cst, xx_ref, uu_ref)

        ############################
        # Update the current solution
        ############################

        xx_temp = np.zeros((constants.NUMBER_OF_STATES, constants.TT))
        uu_temp = np.zeros((constants.NUMBER_OF_INPUTS, constants.TT))

        xx_temp[:, 0] = x0

        for tt in range(constants.TT - 1):
            uu_temp[:, tt] = uu[:, tt, kk] + stepsize * deltau[:, tt, kk]
            xx_temp[:, tt + 1] = dyn.dynamics(xx_temp[:, tt], uu_temp[:, tt])[0]

        xx[:, :, kk + 1] = xx_temp
        uu[:, :, kk + 1] = uu_temp

        ############################
        # Termination condition
        ############################

        print("Iter = {}\t Descent = {:.3e}\t Cost = {:.3e}".format(kk, descent[kk], JJ[kk]))

        if kk%5==0 and kk!=0:
            #Plotting intermediate trajectories
            plots.plot_ref_and_star_trajectories(xx_ref, uu_ref, xx[:, :, kk], uu[:, :, kk])

        if descent[kk] <= term_cond:
            max_iters = kk

            break

    xx_star = xx[:, :, max_iters - 1]
    uu_star = uu[:, :, max_iters - 1]
    uu_star[:, -1] = uu_star[:, -2]  # for plotting purposes

    return xx_star, uu_star
