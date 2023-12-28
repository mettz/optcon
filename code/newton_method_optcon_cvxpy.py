import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np

import constants
import cost as cst
import dynamics as dyn
import plots as plots

max_iters = 20

# Definition of the Armijo parameters
armijo_maxiters = 20
cc = 0.5
beta = 0.7

visu_armijo = True
JJ = np.zeros(max_iters)
lmbd = np.zeros((constants.NUMBER_OF_STATES, constants.TT, max_iters))  # lambdas - costate seq.
descent_arm = np.zeros(max_iters)
deltau = np.zeros((constants.NUMBER_OF_INPUTS, constants.TT, max_iters))  # Du - descent direction
dJ = np.zeros((constants.NUMBER_OF_INPUTS, constants.TT, max_iters)) 

Q = cst.QQt
R = cst.RRt

def armijo_stepsize(xx_ref, uu_ref, xx, uu, delta_u, kk, descent_arm):
    stepsizes = []  # list of stepsizes
    costs_armijo = [] 
    stepsize_0 = 1
    stepsize = stepsize_0

    for ii in range(armijo_maxiters):
        # temp solution update
        print("Armijo iteration {}".format(ii))
        xx_temp = np.zeros((constants.NUMBER_OF_STATES, constants.TT))
        uu_temp = np.zeros((constants.NUMBER_OF_INPUTS, constants.TT))

        xx_temp[:, 0] = xx[:, 0, kk]

        for tt in range(constants.TT - 1):

            uu_temp[:, tt] = uu[:, tt, kk] + stepsize * delta_u[:,tt]
            xx_temp[:, tt + 1] = dyn.dynamics(xx_temp[:, tt], uu_temp[:, tt])[0]

        # temp cost calculation
        JJ_temp = 0

        for tt in range(constants.TT - 1):
            temp_cost = cst.stagecost(xx_temp[:, tt], uu_temp[:, tt], xx_ref[:, tt], uu_ref[:, tt])[0]
            JJ_temp += temp_cost

        temp_cost = cst.termcost(xx_temp[:, -1], xx_ref[:, -1])[0]
        JJ_temp += temp_cost

        stepsizes.append(stepsize)  # save the stepsize
        costs_armijo.append(JJ_temp)  # save the cost associated to the stepsize

        if JJ_temp > JJ[kk] + cc * stepsize * descent_arm:
            # update the stepsize
            stepsize = beta * stepsize
            print("Armijo temp stepsize = {:.3e}".format(stepsize))

            if ii == armijo_maxiters-1:
                print("Armijo stepsize = {:.3e}".format(stepsize))
                if visu_armijo and kk % 10 == 0:
                    plots.armijo_plot(stepsize_0, stepsizes, costs_armijo, descent_arm, JJ, kk, cc, xx[:,0,kk], uu, delta_u, dyn, cst, xx_ref, uu_ref)
                
                return stepsize
            
        else:
            print("Armijo stepsize = {:.3e}".format(stepsize))
            
            if visu_armijo and kk % 10 == 0:
                plots.armijo_plot(stepsize_0, stepsizes, costs_armijo, descent_arm, JJ, kk, cc, xx[:,0,kk], uu, delta_u, dyn, cst, xx_ref, uu_ref)
                
            return stepsize
        
    
        
  

def newton_method_optcon(xx_ref, uu_ref):
    print("Newton method starting...")  # For debugging purposes
    # Step 0 (Initialization): consider an initial guess for the trajectory, so at iteration 0. The initial guess must contain all the trajectory
    
    xx = np.zeros((constants.NUMBER_OF_STATES, constants.TT, max_iters))  # 3x10000 
    uu = np.zeros((constants.NUMBER_OF_INPUTS, constants.TT, max_iters))  # 2x10000

    kk = 0  # Definition of the iterations variable

    # Initialization to zero of the derivatives wrt x and u
    #fx = np.zeros((constants.NUMBER_OF_STATES, constants.NUMBER_OF_STATES, constants.TT))
    #fu = np.zeros((constants.NUMBER_OF_INPUTS, constants.NUMBER_OF_STATES, constants.TT))

    # Initialization to zero of the matrices Q, R, S, q and r
    #qqt = np.zeros((constants.NUMBER_OF_STATES, constants.TT))
    #rrt = np.zeros((constants.NUMBER_OF_INPUTS, constants.TT))

    # Initialization of the trajectory at the first equilibrium point
    xx[:, :, kk] = np.tile(xx_ref[:, 0], (constants.TT, 1)).T
    uu[:, :, kk] = np.tile(uu_ref[:, 0], (constants.TT, 1)).T

    print("Initialization done")  # For debugging purposes

    for kk in range(max_iters - 1):
        JJ[kk] = 0
        print(f"Iteration {kk}")

        # calculate cost
        for tt in range(constants.TT-1):
            temp_cost = cst.stagecost(xx[:,tt, kk], uu[:,tt,kk], xx_ref[:,tt], uu_ref[:,tt])[0]
            JJ[kk] += temp_cost

        temp_cost = cst.termcost(xx[:,-1,kk], xx_ref[:,-1])[0]
        JJ[kk] += temp_cost

        #Solve backward the costate equation to get lmbd, which is used in armijo
        lmbd_temp = cst.termcost(xx[:, constants.TT - 1, kk], xx_ref[:, constants.TT - 1])[1]
        lmbd[:, constants.TT - 1, kk] = lmbd_temp.squeeze()

        print("Solving backward the costate equation")  # For debugging purposes
        for tt in reversed(range(constants.TT - 1)):

            qqt= cst.stagecost(xx[:, tt, kk], uu[:, tt, kk], xx_ref[:, tt], uu_ref[:, tt])[1]
            rrt= cst.stagecost(xx[:, tt, kk], uu[:, tt, kk], xx_ref[:, tt], uu_ref[:, tt])[2]

            fx= dyn.dynamics(xx[:, tt, kk], uu[:, tt, kk])[1]
            fu= dyn.dynamics(xx[:, tt, kk], uu[:, tt, kk])[2]

            lmbd_temp = fx @ lmbd[:, tt + 1, kk][:, None] + qqt  # costate equation
            
            lmbd[:, tt, kk] = lmbd_temp.squeeze() # costate equation
            dJ_temp = fu @ lmbd[:, tt + 1, kk][:, None] + rrt  # gradient of J wrt u
            #deltau_temp = -dJ_temp

            dJ[:, tt, kk] = dJ_temp.squeeze()
            #deltau[:, tt, kk] = deltau_temp.squeeze()

            #descent_arm[kk] += dJ[:, tt, kk].T @ deltau[:, tt, kk] #Calcolarla quando si ha la direzione di discesa

        # Definition of the decision variables (with the state augmentated)
        delta_x = cp.Variable((constants.NUMBER_OF_STATES, constants.TT+1))  # chat gpt: TT + 1
        delta_u = cp.Variable((constants.NUMBER_OF_INPUTS, constants.TT))

        # Definition of the objective function (= cost function) and the constraints
        constraints = []
        cost_function = 0

        for tt in range(constants.TT - 1):
            q = cst.stagecost(xx[:, tt, kk], uu[:, tt, kk], xx_ref[:, tt], uu_ref[:, tt])[1]
            r = cst.stagecost(xx[:, tt, kk], uu[:, tt, kk], xx_ref[:, tt], uu_ref[:, tt])[2]
            A = dyn.dynamics(xx[:, tt, kk], uu[:, tt, kk])[1].T
            B = dyn.dynamics(xx[:, tt, kk], uu[:, tt, kk])[2].T

            # Computation of the stage cost
            cost_function += q.T @ delta_x[:, tt] + r.T @ delta_u[:, tt] + 0.5 * cp.quad_form(delta_x[:, tt], Q) + 0.5 * cp.quad_form(delta_u[:, tt], R)
            
            constraints.append(delta_x[:, tt + 1] == A @ delta_x[:, tt] + B @ delta_u[:, tt])

        # Computation of the terminal cost and initial constraint
        q = cst.termcost(xx[:, constants.TT - 1, kk], xx_ref[:, constants.TT - 1])[1]
        cost_function += q.T @ delta_x[:, constants.TT - 1] + 0.5 * cp.quad_form(delta_x[:, constants.TT - 1], Q)

        constraints.append(delta_x[:, 0] == np.zeros(constants.NUMBER_OF_STATES))
        print("delta_x[:,0]",delta_x[:,0])
        print("shape delta_x[:,0]",delta_x[:,0].shape)
        print("Cost function and constraints computed")  # For debugging purposes

        
        # Definition of the optimization problem
        problem = cp.Problem(cp.Minimize(cost_function), constraints)

        print("Problem defined")  # For debugging purposes
        # Solution of the problem
        problem.solve(verbose=False)
        print("Problem solved")  # For debugging purposes
        # Achievement of the optimal values
        delta_u_star = delta_u.value

        #UPDATE: ho visto dai plot di Armijo che la retta non era tangente al costo. A quel punto ho capito cosa intendeva Sforni con 
        # "calcolate descent_armijo quando avete la direzione di discesa". Quindi l'ho calcolata qui dopo aver ottenuto delta_u_star.
        # ora la retta è costante, però nei plot di Armijo continua ad esserci un comportamento strano.
        for tt in range(constants.TT - 1):
            descent_arm[kk] += dJ[:, tt, kk].T @ delta_u_star[:,tt]

        # STEP 2: Computation of the input sequence
        stepsize = armijo_stepsize(xx_ref, uu_ref, xx, uu, delta_u_star, kk, descent_arm[kk])
        #stepsize = 0.2

        ############################
        # Update the current solution
        ############################

        xx_temp = np.zeros((constants.NUMBER_OF_STATES,constants.TT))
        uu_temp = np.zeros((constants.NUMBER_OF_INPUTS,constants.TT))

        xx_temp[:,0] = xx_ref[:,0]

        for tt in range(constants.TT-1):
            uu[:,tt,kk+1] = uu[:,tt,kk] + stepsize*delta_u_star[:,tt]
            xx[:,tt+1,kk+1] = dyn.dynamics(xx[:,tt,kk+1], uu[:,tt,kk+1])[0]


        ############################
        # Termination condition
        ############################

        print('Cost = {:.3e}'.format(JJ[kk]))

        xx_star = xx[:,:,max_iters-1]
        uu_star = uu[:,:,max_iters-1]
        uu_star[:,-1] = uu_star[:,-2] # for plotting purposes

    return xx_star, uu_star
