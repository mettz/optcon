import cvxpy as cp
import numpy as np
import dynamics as dyn
import cost as cst

tf = 5  # final time in seconds
dt = dyn.dt  # get discretization step from dynamics
ns = dyn.number_of_states
ni = dyn.number_of_inputs
TT = int(tf / dt)  # discrete-time samples


def newton_method_optcon(xx_ref, uu_ref):
    # Step 0 (Initialization): consider an initla guess for the trajectory, so at iteration 0.
    # It must contain all the trajectory (so until TT, time in main 10000), k  times (number of iterations)
    print("Newton method starting...")

    max_iters = 10
    xx = np.ones((ns, TT, max_iters))  # 3x10000
    uu = np.ones((ni, TT, max_iters))  # 2x10000

    # for k=0,1,...
    kk = 0

    fx = np.zeros((ns, ns, TT, max_iters))
    fu = np.zeros((ni, ns, TT, max_iters))

    Qt = np.zeros((ns, ns, TT, max_iters))
    Rt = np.zeros((ni, ni, TT, max_iters))
    St = np.zeros((ni, ns, TT, max_iters))
    qqt = np.zeros((ns, TT, max_iters))
    rrt = np.zeros((ni, TT, max_iters))

    print("Initialization done")

    for kk in range(max_iters):
        print(f"Iteration {kk}")
        if kk == 0:
            # Initialization of the trajectory at the equilibrium point
            xx[:, :, kk] = xx_ref
            xx[:, 0, kk + 1] = xx_ref[:, 0]
            uu[:, :, kk] = uu_ref

        # Evaluate nabla1f, nabla2f, nabla1cost, nabla2cost, nablaTcost and hessians
        for tt in range(TT - 1):  # da 0 a 9999
            qqt[:, tt, kk] = cst.stagecost(xx[:, tt, kk], uu[:, tt, kk], xx_ref[:, tt], uu_ref[:, tt])[1].squeeze()
            rrt[:, tt, kk] = cst.stagecost(xx[:, tt, kk], uu[:, tt, kk], xx_ref[:, tt], uu_ref[:, tt])[2].squeeze()

            fx[:, :, tt, kk] = dyn.dynamics(xx[:, tt, kk], uu[:, tt, kk])[1].squeeze()
            fu[:, :, tt, kk] = dyn.dynamics(xx[:, tt, kk], uu[:, tt, kk])[2].squeeze()

            Qt[:, :, tt, kk], Rt[:, :, tt, kk], St[:, :, tt, kk] = cst.hessian_cost()

        qqt[:, TT - 1, kk] = cst.termcost(xx[:, TT - 1, kk], xx_ref[:, TT - 1])[1].squeeze()
        Qt[:, :, TT - 1, kk] = cst.hessian_term_cost()

        print("Qt,k ; Rt,k ; St,k, qt,k, rt,k computed\n")

        # Define the variables with the state augmentated
        delta_x_tilda = cp.Variable((ns + 1, TT))
        delta_u = cp.Variable((ni, TT))

        # Define the objective function
        obj = 0

        for tt in range(TT - 1):
            q = qqt[:, tt, kk, None]
            r = rrt[:, tt, kk, None]
            Q = Qt[:, :, tt, kk]
            R = Rt[:, :, tt, kk]
            S = St[:, :, tt, kk]
            Q_tilda = np.vstack((np.hstack((np.zeros((1, 1)), q.T)), np.hstack((q, Q))))
            S_tilda = np.hstack((r, S))
            R_tilda = R
            quad1 = cp.quad_form(delta_x_tilda[:, tt, None], Q_tilda)
            quad4 = cp.quad_form(delta_u[:, tt, None], R_tilda)
            obj += 0.5 * (quad1 + quad4)

        q = qqt[:, TT - 1, kk, None]
        Q = Qt[:, :, TT - 1, kk]
        Q_tilda = np.vstack((np.hstack((np.zeros((1, 1)), q.T)), np.hstack((q, Q))))
        obj += 0.5 * cp.quad_form(delta_x_tilda[:, TT - 1, None], Q_tilda)

        # Define the constraints
        constraints = []
        for tt in range(TT - 1):
            A = fx[:, :, tt, kk].T
            B = fu[:, :, tt, kk].T
            A_tilda = np.vstack((np.hstack((np.ones((1, 1)), np.zeros((1, ns)))), np.hstack((np.zeros((ns, 1)), A))))
            B_tilda = np.vstack((np.zeros((1, ni)), B))
            constraints.append(delta_x_tilda[:, tt + 1] == A_tilda @ delta_x_tilda[:, tt] + B_tilda @ delta_u[:, tt])
        constraints.append(delta_x_tilda[:, 0, None] == np.vstack((np.ones((1, 1)), np.zeros((ns, 1)))))

        # Define the problem
        problem = cp.Problem(cp.Minimize(cp.sum(obj)), constraints)

        # Solve the problem
        problem.solve()

        # Get the optimal values
        delta_u_star = delta_u.value
        print("delta_u computed \n")

        # STEP 2: compute the input sequence
        print("Computing the input sequence...")

        stepsize = 0.7
        for tt in range(TT - 1):
            uu[:, tt, kk + 1] = uu[:, tt, kk] + stepsize * delta_u_star[:, tt]
        print("Input sequence computed \n")

        # STEP 3: compute the state sequence
        print("Computing the state sequence...")

        for tt in range(TT - 1):
            xx[:, tt + 1, kk + 1] = dyn.dynamics(xx[:, tt, kk + 1], uu[:, tt, kk + 1])[0]
        print("State sequence computed \n")

    xx_star = xx[:, :, max_iters - 1]
    uu_star = uu[:, :, max_iters - 1]
    # uu_star[:, -1] = uu_star[:, -2]  # for plotting purposes

    for kk in range(max_iters):
        cost = 0
        for tt in range(TT - 1):
            cost += cst.stagecost(xx[:, tt, kk], uu[:, tt, kk], xx_ref[:, tt], uu_ref[:, tt])[0]
            print("delta_u", delta_u[:, tt, kk])
        cost += cst.termcost(xx[:, TT - 1, kk], xx_ref[:, TT - 1])[0]

        print(f"Cost at iteration {kk}: {cost}")

    return xx_star, uu_star
