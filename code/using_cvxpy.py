import cvxpy as cp
import numpy as np
import dynamics as dyn
import cost as cst

tf = 10  # final time in seconds
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

    at = np.zeros((ns, TT, max_iters))
    bt = np.zeros((ni, TT, max_iters))
    fx = np.zeros((ns, ns, TT, max_iters))
    fu = np.zeros((ni, ns, TT, max_iters))
    lT = np.zeros((max_iters))

    fxx = np.zeros((ns, ns, TT, max_iters))
    fuu = np.zeros((ni, ni, TT, max_iters))
    fxu = np.zeros((ni, ns, TT, max_iters))

    Qt = np.zeros((ns, ns, TT, max_iters))
    Rt = np.zeros((ni, ni, TT, max_iters))
    St = np.zeros((ni, ns, TT, max_iters))
    qqt = np.zeros((ns, TT, max_iters))
    rrt = np.zeros((ni, TT, max_iters))

    sigma_t = np.zeros((ni, TT, max_iters))

    delta_u = np.zeros((ni, TT, max_iters))
    delta_x = np.zeros((ns, TT, max_iters))
    print("Initialization done")

    for kk in range(max_iters):
        print(f"Iteration {kk}")
        if kk == 0:
            # Initialization of the trajectory at the equilibrium point
            xx[:, :, kk] = xx_ref
            xx[:, 0, kk + 1] = xx_ref[:, 0]
            uu[:, :, kk] = uu_ref
            delta_x[:, 0, kk] = xx_ref[:, 0]

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

        # Define the variables
        delta_x = cp.Variable((ns, TT))
        delta_u = cp.Variable((ni, TT))

        # Define the objective function
        obj = 0

        for tt in range(TT - 1):
            q = qqt[:, tt, kk]
            r = rrt[:, tt, kk]
            Q = Qt[:, :, tt, kk]
            R = Rt[:, :, tt, kk]
            S = St[:, :, tt, kk]
            obj += cp.matmul(cp.Constant(np.concatenate([q, r]).T), cp.hstack([delta_x[:, tt], delta_u[:, tt]])) + 0.5 * cp.quad_form(
                cp.hstack([delta_x[:, tt], delta_u[:, tt]]), cp.Constant(np.vstack((np.hstack((Q, S.T)), np.hstack((S, R)))))
            )

        q = cp.Constant(qqt[:, TT - 1, kk].T)
        Q = cp.Constant(Qt[:, :, TT - 1, kk])
        obj += 0.5 * cp.quad_form(delta_x[:, TT - 1], Q) + cp.matmul(q, delta_x[:, TT - 1])

        # Define the constraints
        constraints = []
        for tt in range(TT - 1):
            grad1_f = cp.Constant(fx[:, :, tt, kk].T)
            grad2_f = cp.Constant(fu[:, :, tt, kk].T)
            constraints.append(delta_x[:, tt + 1] == cp.matmul(grad1_f, delta_x[:, tt]) + cp.matmul(grad2_f, delta_u[:, tt]))
        constraints.append(delta_x[:, 0] == 0)

        # Define the problem
        problem = cp.Problem(cp.Minimize(obj), constraints)

        # Solve the problem
        problem.solve()

        # Get the optimal values
        delta_x_star = delta_x.value
        delta_u_star = delta_u.value
        print("delta_u and delta_x computed \n")

        # STEP 2: compute the input sequence
        print("Computing the input sequence...")
        print("delta_x", delta_x[:, :, kk])
        print("sigma_t", sigma_t[:, :, kk])

        stepsize = 0.7
        for tt in range(TT - 1):
            uu[:, tt, kk + 1] = uu[:, tt, kk] + stepsize * delta_u_star[:, tt, kk]
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
