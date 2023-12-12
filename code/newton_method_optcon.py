import numpy as np
import dynamics as dyn
import cost as cst

tf = 10  # final time in seconds
dt = dyn.dt  # get discretization step from dynamics
ns = dyn.number_of_states
ni = dyn.number_of_inputs
TT = int(tf / dt)  # discrete-time samples


def newton_method_optcon(xx_ref, uu_ref):
    print("Newton method starting...")
    # Step 0 (Initialization): consider an initial guess for the trajectory, so at iteration 0. It must contain all the trajectory
    max_iters = 10
    xx = np.ones((ns, TT, max_iters))  # 6x10000x10
    print('xx_shape', xx.shape)
    uu = np.ones((ni, TT, max_iters))  # 2x10000x10
    print('uu_shape', uu.shape)
    print("xx_ref", xx_ref.shape)

    kk = 0 # Definition of the iterations variable

    At = np.zeros((ns, ns, TT, max_iters))
    Bt = np.zeros((ns, ni, TT, max_iters))

    at = np.zeros((ns, TT, max_iters))
    bt = np.zeros((ni, TT, max_iters))
    fx = np.zeros((ns, ns, TT, max_iters))
    fu = np.zeros((ni, ns, TT, max_iters))
    lT = np.zeros((max_iters))

    fxx = np.zeros((ns, ns, TT, max_iters))
    fuu = np.zeros((ni, ni, TT, max_iters))
    fxu = np.zeros((ni, ns, TT, max_iters))

    #lmbd = np.zeros((ns, TT, max_iters))  # lambdas - costate seq.

    Qt = np.zeros((ns, ns, TT, max_iters))
    Rt = np.zeros((ni, ni, TT, max_iters))
    St = np.zeros((ni, ns, TT, max_iters))
    qqt = np.zeros((ns, TT, max_iters))
    rrt = np.zeros((ni, TT, max_iters))

    MMt_inv = np.zeros((ni, ni, TT, max_iters))
    mmt = np.zeros((ni, TT, max_iters))

    KK = np.zeros((ni, ns, TT, max_iters))
    PP = np.zeros((ns, ns, TT, max_iters))
    pp = np.zeros((ns, TT, max_iters))
    sigma_t = np.zeros((ni, TT, max_iters))

    delta_u = np.zeros((ni, TT, max_iters)) 
    delta_x = np.zeros((ns, TT, max_iters))
    print("Initialization done")
    
    # xx[:, :, 0] = xx_ref 
    # uu[:, :, 0] = uu_ref
    # delta_x[:, :, 0] = xx_ref
    
    for kk in range(max_iters - 1):
        print(f"Iteration {kk}")
        if kk == 0:
            # Initialization of the trajectory at the equilibrium point
            xx[:, :, kk] = xx_ref
            xx[:, 0, kk + 1] = xx_ref[:, 0]
            uu[:, :, kk] = uu_ref
            delta_x[:, 0, kk] = xx_ref[:, 0]

        # Evaluate nabla1f, nabla2f, nabla1cost, nabla2cost, nablaTcost and hessians
        for tt in range(TT):  # da 0 a 9999
            if tt == (TT - 1):
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

        """ #Solve backward the costate equation 

        lmbd[:, TT-1, kk]=lT[kk] #Inizializzazione
        for tt in reversed(range(TT - 1)):  # integration backward in time

            At[:,:,tt,kk] = fx[:,:,tt,kk].T
            Bt[:,:,tt,kk] = fu[:,:,tt,kk].T

            lmbd[:, tt, kk] = lmbd[:, tt + 1, kk] @ At[:,:,tt,kk] + at[:,tt,kk]
        print("Costate equation solved\n") """

        # Compute for all t in TT-1, Qt,k ; Rt,k ; St,k, qt,k, rt,k ====> USE REGULARIZATION (slide 13)
        print("Computing Qt,k ; Rt,k ; St,k, qt,k, rt,k...")

        for tt in range(TT):
            if tt == (TT - 1):
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
        PP[:, :, -1, kk] = Qt[:, :, TT - 1, kk]
        pp[:, -1, kk] = qqt[:, TT - 1, kk]
        print("Computing P and p solving the Riccati equation...")
        for tt in reversed(range(TT - 1)):
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
        for tt in range(TT - 1):
            # Check positive definiteness (?)

            MMt_inv[:, :, tt, kk] = np.linalg.inv(Rt[:, :, tt, kk] + Bt[:, :, tt, kk].T @ PP[:, :, tt + 1, kk] @ Bt[:, :, tt, kk])
            mmt[:, tt, kk] = rrt[:, tt, kk] + Bt[:, :, tt, kk].T @ pp[:, tt + 1, kk]
            # for other purposes we could add a regularization step here...

            KK[:, :, tt, kk] = -MMt_inv[:, :, tt, kk] @ (Bt[:, :, tt, kk].T @ PP[:, :, tt + 1, kk] @ At[:, :, tt, kk] + St[:, :, tt, kk])
            sigma_t[:, tt, kk] = -MMt_inv[:, :, tt, kk] @ mmt[:, tt, kk]
        print("KK and sigma_t computed \n")

        print("Computing delta_u and delta_x...")
        # Evaluate delta_u
        for tt in range(TT - 1):
            delta_u[:, tt, kk] = KK[:, :, tt, kk] @ delta_x[:, tt, kk] + sigma_t[:, tt, kk]
            delta_x[:, tt + 1, kk] = At[:, :, tt, kk] @ delta_x[:, tt, kk] + Bt[:, :, tt, kk] @ delta_u[:, tt, kk]
        print("delta_u and delta_x computed \n")

        # STEP 2: compute the input sequence
        print("Computing the input sequence...")
        print("delta_x", delta_x[:, :, kk])
        print("sigma_t", sigma_t[:, :, kk])

        stepsize = 0.7
        for tt in range(TT - 1):
            uu[:, tt, kk + 1] = uu[:, tt, kk] + stepsize * delta_u[:, tt, kk]
        print("Input sequence computed \n")

        # STEP 3: compute the state sequence
        print("Computing the state sequence...")

        for tt in range(TT - 1):
            xx[:, tt + 1, kk + 1] = dyn.dynamics(xx[:, tt, kk + 1], uu[:, tt, kk + 1])[0]
        print("State sequence computed \n")

    xx_star = xx[:, :, max_iters - 1]
    uu_star = uu[:, :, max_iters - 1]
    uu_star[:, -1] = uu_star[:, -2]  # For plotting purposes

    for kk in range(max_iters):
        cost = 0
        for tt in range(TT):
            cost += cst.stagecost(xx[:, tt, kk], uu[:, tt, kk], xx_ref[:, tt], uu_ref[:, tt])[0]
            #print("delta_u", delta_u[:, tt, kk])  # For debugging purposes
        cost += cst.termcost(xx[:, TT - 1, kk], xx_ref[:, TT - 1])[0]

        print(f"Cost at iteration {kk}: {cost}")

    return xx_star, uu_star
