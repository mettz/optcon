import numpy as np

import constants
import cost as cst
import dynamics as dyn

from newton_method_optcon_cvxpy import armijo_stepsize  

def newton_method_optcon(xx_ref, uu_ref):
    print("Newton method starting...")
    # Step 0 (Initialization): consider an initial guess for the trajectory, so at iteration 0. It must contain all the trajectory
    max_iters = 40
    xx = np.ones((constants.NUMBER_OF_STATES, constants.TT, max_iters))  # 6x10000x10
    print('xx_shape', xx.shape)
    uu = np.ones((constants.NUMBER_OF_INPUTS, constants.TT, max_iters))  # 2x10000x10
    print('uu_shape', uu.shape)
    print("xx_ref", xx_ref.shape)

    kk = 0 # Definition of the iterations variable

    At = np.zeros((constants.NUMBER_OF_STATES, constants.NUMBER_OF_STATES, constants.TT, max_iters))
    Bt = np.zeros((constants.NUMBER_OF_STATES, constants.NUMBER_OF_INPUTS, constants.TT, max_iters))

    at = np.zeros((constants.NUMBER_OF_STATES, constants.TT, max_iters))
    bt = np.zeros((constants.NUMBER_OF_INPUTS, constants.TT, max_iters))
    fx = np.zeros((constants.NUMBER_OF_STATES, constants.NUMBER_OF_STATES, constants.TT, max_iters))
    fu = np.zeros((constants.NUMBER_OF_INPUTS, constants.NUMBER_OF_STATES, constants.TT, max_iters))
    lT = np.zeros((max_iters))

    fxx = np.zeros((constants.NUMBER_OF_STATES, constants.NUMBER_OF_STATES, constants.TT, max_iters))
    fuu = np.zeros((constants.NUMBER_OF_INPUTS, constants.NUMBER_OF_INPUTS, constants.TT, max_iters))
    fxu = np.zeros((constants.NUMBER_OF_INPUTS, constants.NUMBER_OF_STATES, constants.TT, max_iters))

    lmbd = np.zeros((constants.NUMBER_OF_STATES, constants.TT, max_iters))  # lambdas - costate seq.
    dJ = np.zeros((constants.NUMBER_OF_INPUTS, constants.TT, max_iters)) 
    descent_arm = np.zeros(max_iters)

    JJ = np.zeros(max_iters)

    Qt = np.zeros((constants.NUMBER_OF_STATES, constants.NUMBER_OF_STATES, constants.TT, max_iters))
    Rt = np.zeros((constants.NUMBER_OF_INPUTS, constants.NUMBER_OF_INPUTS, constants.TT, max_iters))
    St = np.zeros((constants.NUMBER_OF_INPUTS, constants.NUMBER_OF_STATES, constants.TT, max_iters))
    qqt = np.zeros((constants.NUMBER_OF_STATES, constants.TT, max_iters))
    rrt = np.zeros((constants.NUMBER_OF_INPUTS, constants.TT, max_iters))

    MMt_inv = np.zeros((constants.NUMBER_OF_INPUTS, constants.NUMBER_OF_INPUTS, constants.TT, max_iters))
    mmt = np.zeros((constants.NUMBER_OF_INPUTS, constants.TT, max_iters))

    KK = np.zeros((constants.NUMBER_OF_INPUTS, constants.NUMBER_OF_STATES, constants.TT, max_iters))
    PP = np.zeros((constants.NUMBER_OF_STATES, constants.NUMBER_OF_STATES, constants.TT, max_iters))
    pp = np.zeros((constants.NUMBER_OF_STATES, constants.TT, max_iters))
    sigma_t = np.zeros((constants.NUMBER_OF_INPUTS, constants.TT, max_iters))

    delta_u = np.zeros((constants.NUMBER_OF_INPUTS, constants.TT, max_iters)) 
    delta_x = np.zeros((constants.NUMBER_OF_STATES, constants.TT, max_iters))

    kk=0

    xx[:, :, kk] = np.tile(xx_ref[:, 0], (constants.TT, 1)).T
    uu[:, :, kk] = np.tile(uu_ref[:, 0], (constants.TT, 1)).T
    print("Initialization done")
    
    # xx[:, :, 0] = xx_ref 
    # uu[:, :, 0] = uu_ref
    # delta_x[:, :, 0] = xx_ref
    
    for kk in range(max_iters - 1):
        print(f"Iteration {kk}")
        JJ[kk] = 0

        # calculate cost
        for tt in range(constants.TT-1):
            temp_cost = cst.stagecost(xx[:,tt, kk], uu[:,tt,kk], xx_ref[:,tt], uu_ref[:,tt])[0]
            JJ[kk] += temp_cost

        temp_cost = cst.termcost(xx[:,-1,kk], xx_ref[:,-1])[0]
        JJ[kk] += temp_cost

        # Evaluate nabla1f, nabla2f, nabla1cost, nabla2cost, nablaTcost and hessians
        for tt in range(constants.TT):  # da 0 a 9999
            if tt == (constants.TT - 1):
                lT[kk] = cst.termcost(xx[:, tt, kk], xx_ref[:, tt])[0]  # VETTORE, per la costate equation
            else:
                # at[:,tt,kk], bt[:,tt,kk] = cst.stagecost(xx[:, tt, kk], uu[:, tt, kk], xx_ref[:, tt], uu_ref[:, tt])[1:].squeeze()
                at[:, tt, kk] = cst.stagecost(xx[:, tt, kk], uu[:, tt, kk], xx_ref[:, tt], uu_ref[:, tt])[1].squeeze()
                bt[:, tt, kk] = cst.stagecost(xx[:, tt, kk], uu[:, tt, kk], xx_ref[:, tt], uu_ref[:, tt])[2].squeeze()

                # fx[:,tt,kk], fu[:,tt,kk] = dyn.dynamics(xx[:, tt, kk], uu[:, tt, kk])[1:]
                fx[:, :, tt, kk] = dyn.dynamics(xx[:, tt, kk], uu[:, tt, kk])[1].squeeze()
                fu[:, :, tt, kk] = dyn.dynamics(xx[:, tt, kk], uu[:, tt, kk])[2].squeeze()

                fxx[:, :, tt, kk], fuu[:, :, tt, kk], fxu[:, :, tt, kk] = cst.hessian_cost()

        print("Evaluation of nabla1f, nabla2f, nabla1cost, nabla2cost, nablaTcost and hessians done\n")

        #Solve backward the costate equation 

        lmbd_temp = cst.termcost(xx[:, constants.TT - 1, kk], xx_ref[:, constants.TT - 1])[1]
        lmbd[:, constants.TT - 1, kk] = lmbd_temp.squeeze()

        for tt in reversed(range(constants.TT - 1)):  # integration backward in time

            At[:,:,tt,kk] = fx[:,:,tt,kk].T
            Bt[:,:,tt,kk] = fu[:,:,tt,kk].T

            lmbd_temp = At[:,:,tt,kk].T @ lmbd[:, tt + 1, kk][:, None] + at[:,tt,kk]  # costate equation
            print("lmbd_temp", lmbd_temp.shape)
            lmbd[:, tt, kk] = lmbd_temp.squeeze()# costate equation

            dJ_temp = Bt[:,:,tt,kk].T @ lmbd[:, tt + 1, kk][:, None] + bt[:,tt,kk]  # gradient of J wrt u
            dJ[:, tt, kk] = dJ_temp

        print("Costate equation solved\n") 

        # Compute for all t in TT-1, Qt,k ; Rt,k ; St,k, qt,k, rt,k ====> USE REGULARIZATION (slide 13)
        print("Computing Qt,k ; Rt,k ; St,k, qt,k, rt,k...")

        for tt in range(constants.TT):
            if tt == (constants.TT - 1):
                Qt[:, :, tt, kk] = fxx[:, :, tt, kk]
                qqt[:, tt, kk] = at[:, tt, kk]
            else:
                Qt[:, :, tt, kk] = fxx[:, :, tt, kk]
                Rt[:, :, tt, kk] = fuu[:, :, tt, kk]
                St[:, :, tt, kk] = fxu[:, :, tt, kk]
                qqt[:, tt, kk] = at[:, tt, kk]
                rrt[:, tt, kk] = bt[:, tt, kk]
        print("Qt,k ; Rt,k ; St,k, qt,k, rt,k computed\n")

        # Now compute the descent direction, solving the minimization problem to get delta_x and delta_u.
        # It's an affine LQ problem.
        print("Computing the descent direction...")
        PP[:, :, -1, kk] = Qt[:, :, constants.TT - 1, kk]
        pp[:, -1, kk] = qqt[:, constants.TT - 1, kk]
        print("Computing P and p solving the Riccati equation...")
        for tt in reversed(range(constants.TT - 1)):
            MMt_inv[:, :, tt, kk] = np.linalg.inv(Rt[:, :, tt, kk] + Bt[:, :, tt, kk].T @ PP[:, :, tt + 1, kk] @ Bt[:, :, tt, kk])
            mmt[:, tt, kk] = rrt[:, tt, kk] + Bt[:, :, tt, kk].T @ pp[:, tt + 1, kk]

            PP[:, :, tt, kk] = (
                At[:, :, tt, kk].T @ PP[:, :, tt + 1, kk] @ At[:, :, tt, kk]
                - (Bt[:, :, tt, kk].T @ PP[:, :, tt + 1, kk] @ At[:, :, tt, kk] + St[:, :, tt, kk]).T
                @ MMt_inv[:, :, tt, kk]
                @ (Bt[:, :, tt, kk].T @ PP[:, :, tt + 1, kk] @ At[:, :, tt, kk] + St[:, :, tt, kk])
                + Qt[:, :, tt, kk]
            )
            pp[:, tt, kk] = (
                At[:, :, tt, kk].T @ pp[:, tt + 1, kk]
                - (Bt[:, :, tt, kk].T @ PP[:, :, tt + 1, kk] @ At[:, :, tt, kk] + St[:, :, tt, kk]).T @ MMt_inv[:, :, tt, kk] @ mmt[:, tt, kk]
                + qqt[:, tt, kk]
            )
        print("P and p computed\n")

        print("Computing KK and sigma_t...")
        # Evaluate KK
        for tt in range(constants.TT - 1):
            # Check positive definiteness (?)

            MMt_inv[:, :, tt, kk] = np.linalg.inv(Rt[:, :, tt, kk] + Bt[:, :, tt, kk].T @ PP[:, :, tt + 1, kk] @ Bt[:, :, tt, kk])
            mmt[:, tt, kk] = rrt[:, tt, kk] + Bt[:, :, tt, kk].T @ pp[:, tt + 1, kk]
            # for other purposes we could add a regularization step here...

            KK[:, :, tt, kk] = -MMt_inv[:, :, tt, kk] @ (Bt[:, :, tt, kk].T @ PP[:, :, tt + 1, kk] @ At[:, :, tt, kk] + St[:, :, tt, kk])
            sigma_t[:, tt, kk] = -MMt_inv[:, :, tt, kk] @ mmt[:, tt, kk]

        print("KK and sigma_t computed \n")

        print("Computing delta_u and delta_x...")
        # Evaluate delta_u
        for tt in range(constants.TT - 1):
            delta_u[:, tt, kk] = KK[:, :, tt, kk] @ delta_x[:, tt, kk] + sigma_t[:, tt, kk]
            delta_x[:, tt + 1, kk] = At[:, :, tt, kk] @ delta_x[:, tt, kk] + Bt[:, :, tt, kk] @ delta_u[:, tt, kk]

            descent_arm[kk] += dJ[:, tt, kk].T @ delta_u[:,tt,kk]
        print("delta_u and delta_x computed \n")

        # STEP 2: compute the input sequence
        print("Computing the input sequence...")

        stepsize = armijo_stepsize(xx_ref, uu_ref, xx, uu, delta_u, kk, descent_arm[kk])

        xx_temp = np.zeros((constants.NUMBER_OF_STATES, constants.TT))
        uu_temp = np.zeros((constants.NUMBER_OF_INPUTS, constants.TT))

        xx_temp[:, 0] = xx_ref[:, 0]

        for tt in range(constants.TT - 1):
            uu_temp[:, tt] = uu[:, tt, kk] + stepsize * delta_u[:, tt, kk]
            xx_temp[:, tt + 1] = dyn.dynamics(xx_temp[:, tt], uu_temp[:, tt])[0]

        xx[:, :, kk + 1] = xx_temp
        uu[:, :, kk + 1] = uu_temp

        xx_star = xx[:, :, max_iters - 1]
        uu_star = uu[:, :, max_iters - 1]
        uu_star[:, -1] = uu_star[:, -2]  # for plotting purposes

    return xx_star, uu_star
