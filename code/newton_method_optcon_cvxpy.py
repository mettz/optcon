import cvxpy as cp
import numpy as np
import dynamics as dyn
import cost as cst

import matplotlib.pyplot as plt
import plots as plots

tf = 5  # final time in seconds
dt = dyn.dt  # get discretization step from dynamics
ns = dyn.number_of_states  # get number of states from dynamics
ni = dyn.number_of_inputs  # get number of inputs from dynamics
TT = int(tf / dt)  # discrete-time samples
max_iters = 10

armijo_maxiters = 20
cc = 0.5
beta = 0.7

visu_armijo = False
JJ = np.zeros((max_iters, 1))
lmbd = np.zeros((ns, TT, max_iters))  # lambdas - costate seq.
descent_arm = np.zeros(max_iters)
deltau = np.zeros((ni, TT, max_iters))  # Du - descent direction
dJ = np.zeros((ni, TT, max_iters)) 

Q = np.diag([0.1, 0.1, 0.1])
R = np.diag([1, 1])

def armijo_stepsize(xx_ref, uu_ref, xx, uu, delta_u, kk, descent_arm):
    stepsizes = []  # list of stepsizes
    costs_armijo = [] 
    stepsize_0 = 0.7
    stepsize = stepsize_0

    for ii in range(armijo_maxiters):
        # temp solution update
        print("Armijo iteration {}".format(ii))
        xx_temp = np.zeros((ns, TT))
        uu_temp = np.zeros((ni, TT))

        xx_temp[:, 0] = xx[:, 0, kk]

        for tt in range(TT - 1):

            uu_temp[:, tt] = uu[:, tt, kk] + stepsize * delta_u[:,tt]
            xx_temp[:, tt + 1] = dyn.dynamics(xx_temp[:, tt], uu_temp[:, tt])[0]

        # temp cost calculation
        JJ_temp = 0

        for tt in range(TT - 1):
            temp_cost = cst.stagecost(xx_temp[:, tt], uu_temp[:, tt], xx_ref[:, tt], uu_ref[:, tt])[0]
            JJ_temp += temp_cost

        temp_cost = cst.termcost(xx_temp[:, -1], xx_ref[:, -1])[0]
        JJ_temp += temp_cost

        #stepsizes.append(stepsize)  # save the stepsize
        #costs_armijo.append(np.min([JJ_temp, 100 * JJ[kk]]))  # save the cost associated to the stepsize

        if JJ_temp > JJ[kk] + cc * stepsize * descent_arm:
            # update the stepsize
            stepsize = beta * stepsize
            print("Armijo temp stepsize = {:.3e}".format(stepsize))
        else:
            if visu_armijo and kk % 10 == 0:
                plots.armijo_plot(stepsize_0, stepsizes, costs_armijo, descent_arm, JJ, kk, cc, ns, ni, TT, xx[:,0,kk], uu, delta_u, dyn, cst, xx_ref, uu_ref)
                print("Armijo stepsize = {:.3e}".format(stepsize))
                break


def newton_method_optcon(xx_ref, uu_ref):
    print("Newton method starting...")  # For debugging purposes
    # Step 0 (Initialization): consider an initial guess for the trajectory, so at iteration 0. The initial guess must contain all the trajectory
    
    xx = np.zeros((ns, TT, max_iters))  # 3x10000 -> 6x10000
    uu = np.zeros((ni, TT, max_iters))  # 2x10000

    kk = 0  # Definition of the iterations variable

    # Initialization to zero of the derivatives wrt x and u
    fx = np.zeros((ns, ns, TT))
    fu = np.zeros((ni, ns, TT))

    # Initialization to zero of the matrices Q, R, S, q and r
    qqt = np.zeros((ns, TT))
    rrt = np.zeros((ni, TT))

    print("Initialization done")  # For debugging purposes

    for kk in range(max_iters - 1):
        print(f"Iteration {kk}")
        if kk == 0:
            # Initialization of the trajectory at the first equilibrium point
            xx[:, :, kk] = np.tile(xx_ref[:, 0], (TT, 1)).T
            uu[:, :, kk] = np.tile(uu_ref[:, 0], (TT, 1)).T

        # Evaluate nabla1f, nabla2f, nabla1cost, nabla2cost, nablaTcost and hessians
        # from 0 to T - 2 because in python T - 1 will be our T
        print("Computing derivatives...")  # For debugging purposes
        for tt in range(TT - 1):
            JJ[kk] += cst.stagecost(xx[:, tt, kk], uu[:, tt, kk], xx_ref[:, tt], uu_ref[:, tt])[0].squeeze()
            qqt[:, tt] = cst.stagecost(xx[:, tt, kk], uu[:, tt, kk], xx_ref[:, tt], uu_ref[:, tt])[1].squeeze()
            rrt[:, tt] = cst.stagecost(xx[:, tt, kk], uu[:, tt, kk], xx_ref[:, tt], uu_ref[:, tt])[2].squeeze()


            fx[:, :, tt] = dyn.dynamics(xx[:, tt, kk], uu[:, tt, kk])[1].squeeze()
            fu[:, :, tt] = dyn.dynamics(xx[:, tt, kk], uu[:, tt, kk])[2].squeeze()

        
        JJ[kk] += cst.termcost(xx[:, TT - 1, kk], xx_ref[:, TT - 1])[0].squeeze()
        qqt[:, TT - 1] = cst.termcost(xx[:, TT - 1, kk], xx_ref[:, TT - 1])[1].squeeze()

        print("Derivatives computed")  # For debugging purposes

        #Solve backward the costate equation to get lmbd, which is used in armijo
        lmbd_temp = cst.termcost(xx[:, TT - 1, kk], xx_ref[:, TT - 1])[1]
        lmbd[:, TT - 1, kk] = lmbd_temp.squeeze()

        print("Solving backward the costate equation")  # For debugging purposes
        for tt in reversed(range(TT - 1)):
        
            lmbd[:, tt, kk][:,None] = fx[:, :, tt] @ lmbd[:, tt + 1, kk][:, None] + qqt[:, tt, None]  # costate equation
            dJ_temp = fu[:, :, tt] @ lmbd[:, tt + 1, kk][:, None] + rrt[:, tt, None]  # gradient of J wrt u
            deltau_temp = -dJ_temp

            dJ[:, tt, kk] = dJ_temp.squeeze()
            deltau[:, tt, kk] = deltau_temp.squeeze()

            descent_arm[kk] += dJ[:, tt, kk].T @ deltau[:, tt, kk] #Calcolarla quando si ha la direzione di discesa

        # Definition of the decision variables (with the state augmentated)
        delta_x = cp.Variable((ns, TT))  # chat gpt: TT + 1
        delta_u = cp.Variable((ni, TT))

        # Definition of the objective function (= cost function)
        cost_function = 0

        for tt in range(TT - 1):
            q = qqt[:, tt] #Occhio alle dimensioni e ai trasposti. Usare il codice della lezione
            r = rrt[:, tt]

            # Computation of the stage cost
            cost_function += q @ delta_x[:, tt, None] + r @ delta_u[:, tt, None] + 0.5 * cp.quad_form(delta_x[:, tt], Q) + 0.5 * cp.quad_form(delta_u[:, tt], R)

        # Computation of the terminal cost
        q = qqt[:, TT - 1]
        cost_function += q @ delta_x[:, TT - 1] + 0.5 * cp.quad_form(delta_x[:, TT - 1], Q)

        print("Cost function computed")  # For debugging purposes

        # Definition of the constraints (dynamics of the system)
        constraints = []
        
        for tt in range(TT - 1):
            A = fx[:, :, tt].T
            B = fu[:, :, tt].T
            constraints.append(delta_x[:, tt + 1] == A @ delta_x[:, tt] + B @ delta_u[:, tt])
        constraints.append(delta_x[:, 0] == np.zeros(ns))

        # Definition of the optimization problem
        problem = cp.Problem(cp.Minimize(cost_function), constraints)

        print("Problem defined")  # For debugging purposes
        # Solution of the problem
        problem.solve(verbose=False)
        print("Problem solved")  # For debugging purposes
        # Achievement of the optimal values
        delta_u_star = delta_u.value

        # STEP 2: Computation of the input sequence
        #stepsize = armijo_stepsize(xx_ref, uu_ref, xx, uu, delta_u_star, kk, descent_arm[kk])
        stepsize = 0.2
        for tt in range(TT - 1):
            uu[:, tt, kk + 1] = uu[:, tt, kk] + stepsize * delta_u_star[:, tt]

        # STEP 3: Computation of the state sequence
        xx[:, 0, kk + 1] = xx_ref[:, 0]
        for tt in range(TT - 1):
            xx[:, tt + 1, kk + 1] = dyn.dynamics(xx[:, tt, kk + 1], uu[:, tt, kk + 1])[0]

        # states = ["V", "beta", "psi_dot"]
        # inputs = ["delta", "Fx"]
        # tt_hor = np.linspace(0, tf, TT)
        # plt.figure()
        # plt.clf()
        # plt.title("Trajectory following")
        # for i in range(np.size(states)):
        #     plt.subplot(3, 2, 1 + i)
        #     plt.plot(tt_hor, xx_ref[i, :], label=f"Reference curve {states[i]}")
        #     plt.plot(tt_hor, xx[i, :, kk + 1], label=f"State {states[i]}")
        #     plt.grid()
        #     plt.legend()

        # for i in range(np.size(inputs)):
        #     plt.subplot(3, 2, 4 + i)
        #     plt.plot(tt_hor, uu_ref[i, :], label=f"Reference curve {inputs[i]}")
        #     plt.plot(tt_hor, uu[i, :, kk + 1], label=f"State {inputs[i]}")

        #     plt.grid()
        #     plt.legend()
        # plt.show()

    xx_star = xx[:, :, max_iters - 1]
    uu_star = uu[:, :, max_iters - 1]
    uu_star[:, -1] = uu_star[:, -2]  # for plotting purposes

    return xx_star, uu_star
