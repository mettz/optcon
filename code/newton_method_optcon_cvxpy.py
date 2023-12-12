import cvxpy as cp
import numpy as np
import dynamics as dyn
import cost as cst

tf = 5  # final time in seconds
dt = dyn.dt  # get discretization step from dynamics
ns = dyn.number_of_states # get number of states from dynamics
ni = dyn.number_of_inputs # get number of inputs from dynamics
TT = int(tf / dt)  # discrete-time samples


def newton_method_optcon(xx_ref, uu_ref):
    print("Newton method starting...") # For debugging purposes
    # Step 0 (Initialization): consider an initial guess for the trajectory, so at iteration 0. The initial guess must contain all the trajectory
    max_iters = 10
    xx = np.ones((ns, TT, max_iters))  # 3x10000 -> 6x10000
    uu = np.ones((ni, TT, max_iters))  # 2x10000
       
    kk = 0 # Definition of the iterations variable

    # Initialization to zero of the derivatives wrt x and u
    fx = np.zeros((ns, ns, TT, max_iters))
    fu = np.zeros((ni, ns, TT, max_iters))

    # Initialization to zero of the matrices Q, R, S, q and r
    Qt = np.zeros((ns, ns, TT, max_iters))
    Rt = np.zeros((ni, ni, TT, max_iters))
    St = np.zeros((ni, ns, TT, max_iters))
    qqt = np.zeros((ns, TT, max_iters))
    rrt = np.zeros((ni, TT, max_iters))

    print("Initialization done") # For debugging purposes

    for kk in range(max_iters):
        print(f"Iteration {kk}")
        if kk == 0:
            # Initialization of the trajectory at the first equilibrium point
            xx[:, :, kk] = xx_ref
            xx[:, 0, kk + 1] = xx_ref[:, 0]
            uu[:, :, kk] = uu_ref

        # Evaluate nabla1f, nabla2f, nabla1cost, nabla2cost, nablaTcost and hessians
        for tt in range(TT):  # da 0 a 9999 QUI C'ERA TT-1 PRIMA
            qqt[:, tt, kk] = cst.stagecost(xx[:, tt, kk], uu[:, tt, kk], xx_ref[:, tt], uu_ref[:, tt])[1].squeeze()
            rrt[:, tt, kk] = cst.stagecost(xx[:, tt, kk], uu[:, tt, kk], xx_ref[:, tt], uu_ref[:, tt])[2].squeeze()

            fx[:, :, tt, kk] = dyn.dynamics(xx[:, tt, kk], uu[:, tt, kk])[1].squeeze()
            fu[:, :, tt, kk] = dyn.dynamics(xx[:, tt, kk], uu[:, tt, kk])[2].squeeze()

            Qt[:, :, tt, kk], Rt[:, :, tt, kk], St[:, :, tt, kk] = cst.hessian_cost()

        qqt[:, TT - 1, kk] = cst.termcost(xx[:, TT - 1, kk], xx_ref[:, TT - 1])[1].squeeze()
        Qt[:, :, TT - 1, kk] = cst.hessian_term_cost()

        print("Qt,k ; Rt,k ; St,k, qt,k, rt,k computed\n")

        # Definition of the decision variables (with the state augmentated)
        #delta_x_tilda = cp.Variable((ns + 1, TT))
        delta_x_tilda = cp.Variable((ns + 1, TT)) #chat gpt: TT + 1 
        delta_u = cp.Variable((ni, TT))

        # Definition of the objective function (= cost function)
        cost_function = 0

        for tt in range(TT): # Perchè TT-1 ? Era così...
            q = qqt[:, tt, kk, None]
            r = rrt[:, tt, kk, None]
            Q = Qt[:, :, tt, kk]
            R = Rt[:, :, tt, kk]
            S = St[:, :, tt, kk]
            Q_tilda = np.vstack((np.hstack((np.zeros((1, 1)), q.T)), np.hstack((q, Q))))
            S_tilda = np.hstack((r, S))
            R_tilda = R
            '''quad1 = cp.quad_form(delta_x_tilda[:, tt, None], Q_tilda)  
            quad4 = cp.quad_form(delta_u[:, tt, None], R_tilda)'''
            # Computation of the stage cost
            cost_function += 0.5 * (cp.quad_form(delta_x_tilda[:, tt, None], Q_tilda) + cp.quad_form(delta_u[:, tt, None], R_tilda))

        # Computation of the terminal cost
        q = qqt[:, TT - 1, kk, None]
        Q = Qt[:, :, TT - 1, kk]
        Q_tilda = np.vstack((np.hstack((np.zeros((1, 1)), q.T)), np.hstack((q, Q))))
        cost_function += 0.5 * cp.quad_form(delta_x_tilda[:, TT - 1, None], Q_tilda)

        # Definition of the constraints (dynamics of the system)
        constraints = []
        for tt in range(TT - 1):
            A = fx[:, :, tt, kk].T
            B = fu[:, :, tt, kk].T
            A_tilda = np.vstack((np.hstack((np.ones((1, 1)), np.zeros((1, ns)))), np.hstack((np.zeros((ns, 1)), A))))
            B_tilda = np.vstack((np.zeros((1, ni)), B))
            constraints.append(delta_x_tilda[:, tt + 1] == A_tilda @ delta_x_tilda[:, tt] + B_tilda @ delta_u[:, tt])
        constraints.append(delta_x_tilda[:, 0, None] == np.vstack((np.ones((1, 1)), np.zeros((ns, 1)))))

        # Definition of the optimization problem
        problem = cp.Problem(cp.Minimize(cp.sum(cost_function)), constraints)

        # Solution of the problem
        problem.solve()

        # Achievement of the optimal values
        delta_u_star = delta_u.value
        print("delta_u computed \n") # For debugging purposes

        # STEP 2: Computation of the input sequence
        print("Computing the input sequence...") # For debugging purposes

        stepsize = 0.7
        for tt in range(TT - 1):
            uu[:, tt, kk + 1] = uu[:, tt, kk] + stepsize * delta_u_star[:, tt]
        print("Input sequence computed \n") # For debugging purposes

        # STEP 3: Computation of the state sequence
        print("Computing the state sequence...") # For debugging purposes

        for tt in range(TT-1):
            xx[:, tt + 1, kk + 1] = dyn.dynamics(xx[:, tt, kk + 1], uu[:, tt, kk + 1])[0]
        print("State sequence computed \n")

    xx_star = xx[:, :, max_iters - 1]
    uu_star = uu[:, :, max_iters - 1]
    uu_star[:, -1] = uu_star[:, -2]  # for plotting purposes

    for kk in range(max_iters):
        cost = 0
        for tt in range(TT):
            cost += cst.stagecost(xx[:, tt, kk], uu[:, tt, kk], xx_ref[:, tt], uu_ref[:, tt])[0]
            print("delta_u", delta_u[:, tt, kk])
        cost += cst.termcost(xx[:, TT - 1, kk], xx_ref[:, TT - 1])[0]

        print(f"Cost at iteration {kk}: {cost}")

    return xx_star, uu_star
