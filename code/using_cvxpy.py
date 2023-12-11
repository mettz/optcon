import cvxpy as cp
import numpy as np
import dynamics as dyn
import cost as cst

import numpy as np
import dynamics as dyn
import cost as cst
import time
import cvxpy as cp

def newton_method_optcon_cvxpy (xx_ref, uu_ref):
    #Step 0 (Initialization): consider an initla guess for the trajectory, so at iteration 0.
                            #It must contain all the trajectory (so until TT, time in main 10000), k  times (number of iterations)
    print("Newton method starting...")
    tf = 10  # final time in seconds
    dt = dyn.dt  # get discretization step from dynamics
    ns = dyn.number_of_states
    ni = dyn.number_of_inputs
    TT = int(tf / dt)  # discrete-time samples

    max_iters = 10
    xx = np.ones((ns, TT, max_iters)) # 3x10000
    uu = np.ones((ni, TT, max_iters)) # 2x10000

    #for k=0,1,...
    kk=0
    At=np.zeros((ns,ns,TT,max_iters))
    Bt=np.zeros((ns,ni,TT,max_iters))

    at=np.zeros((ns,TT,max_iters))
    bt=np.zeros((ni,TT,max_iters))
    fx=np.zeros((ns, ns,TT,max_iters))
    fu=np.zeros((ni, ns, TT,max_iters))
    lT=np.zeros((max_iters)) 
    
    fxx=np.zeros((ns, ns, TT, max_iters))
    fuu=np.zeros((ni, ni, TT, max_iters))
    fxu=np.zeros((ni, ns, TT, max_iters))
    
    lmbd = np.zeros((ns, TT, max_iters))  # lambdas - costate seq.

    Qt = np.zeros((ns, ns, TT, max_iters))
    Rt = np.zeros((ni, ni, TT, max_iters))
    St = np.zeros((ni, ns, TT, max_iters))
    qqt = np.zeros((ns, TT, max_iters))
    rrt = np.zeros((ni, TT, max_iters))

    MMt_inv = np.zeros((ni, ni, TT, max_iters))
    mmt = np.zeros((ni, TT, max_iters))

    KK = np.zeros((ni, ns, TT, max_iters))
    PP = np.zeros((ns, ns, TT, max_iters))
    pp = np.zeros((ns,TT,max_iters))
    sigma_t = np.zeros((ni,TT,max_iters))

    delta_u_star = np.zeros((ni, TT-1))
    delta_x_star = np.zeros((ns, TT-1))
    print("Initialization done")

    for kk in range(max_iters-1):
        print(f"Iteration {kk}")
        if kk==0:
           #Initialization of the trajectory at the equilibrium point
            xx[:, :, kk] = xx_ref-0.1
            uu[:, :, kk] = uu_ref-0.1
        
        #Evaluate nabla1f, nabla2f, nabla1cost, nabla2cost, nablaTcost and hessians
        for tt in range(TT): #da 0 a 9999
            if tt == TT:
                lT[kk]=cst.termcost(xx[:, tt, kk],xx_ref[:, tt, kk])[1] #VETTORE, per la costate equation
            else:
                #at[:,tt,kk], bt[:,tt,kk] = cst.stagecost(xx[:, tt, kk], uu[:, tt, kk], xx_ref[:, tt], uu_ref[:, tt])[1:].squeeze() 
                at[:,tt,kk]=cst.stagecost(xx[:, tt, kk], uu[:, tt, kk], xx_ref[:, tt], uu_ref[:, tt])[1].squeeze()
                bt[:,tt,kk]=cst.stagecost(xx[:, tt, kk], uu[:, tt, kk], xx_ref[:, tt], uu_ref[:, tt])[2].squeeze()
                

                #fx[:,tt,kk], fu[:,tt,kk] = dyn.dynamics(xx[:, tt, kk], uu[:, tt, kk])[1:]
                fx[:,:,tt,kk]=dyn.dynamics(xx[:, tt, kk], uu[:, tt, kk])[1].squeeze()
                fu[:,:,tt,kk]=dyn.dynamics(xx[:, tt, kk], uu[:, tt, kk])[2].squeeze()

                fxx[:,:,tt,kk], fuu[:,:,tt,kk], fxu[:,:,tt,kk] = cst.hessian_cost()

        print("Evaluation of nabla1f, nabla2f, nabla1cost, nabla2cost, nablaTcost and hessians done\n")
        

        #Solve backward the costate equation 

        lmbd[:, TT-1, kk]=lT[kk] #Inizializzazione
        for tt in reversed(range(TT - 1)):  # integration backward in time

            At[:,:,tt,kk] = fx[:,:,tt,kk].T
            Bt[:,:,tt,kk] = fu[:,:,tt,kk].T

            lmbd[:, tt, kk] = lmbd[:, tt + 1, kk] @ At[:,:,tt,kk] + at[:,tt,kk]
        print("Costate equation solved\n")

        #Compute for all t in TT-1, Qt,k ; Rt,k ; St,k, qt,k, rt,k ====> USE REGULARIZATION (slide 13)
        print("Computing Qt,k ; Rt,k ; St,k, qt,k, rt,k...")
        
        for tt in range(TT):
          if tt==TT:
            Qt[:,:,tt,kk] = fxx[:,:,tt,kk]
            qqt[:,tt,kk] = at[:,tt,kk]
          else:
            Qt[:,:,tt,kk] = fxx[:,:,tt,kk]
            Rt[:,:,tt,kk] = fuu[:,:,tt,kk]
            St[:,:,tt,kk] = fxu[:,:,tt,kk]
            qqt[:,tt,kk] = at[:,tt,kk]
            rrt[:,tt,kk] = bt[:,tt,kk]
        print("Qt,k ; Rt,k ; St,k, qt,k, rt,k computed\n")

        #Now solve the minimization problem to get delta_u, using cvxpy. It's an affine LQ problem.
        

        # Define the variables
        delta_x = cp.Variable((ns, TT-1))
        delta_u = cp.Variable((ni, TT-1))

        # Define the objective function
        obj = 0
       
        for tt in range(TT-1):
            grad1_ell = cp.Constant(at[:, tt, kk])
            grad2_ell = cp.Constant(bt[:, tt, kk])
            Q = cp.Constant(Qt[:, :, tt, kk])
            R = cp.Constant(Rt[:, :, tt, kk])
            S = cp.Constant(St[:, :, tt, kk])
            obj += cp.quad_form(cp.vstack([delta_x[:, tt], delta_u[:, tt]]), cp.vstack([Q, S.T, S, R])) + cp.matmul(cp.vstack([grad1_ell, grad2_ell]), cp.vstack([delta_x[:, tt], delta_u[:, tt]].T))

        gradT_ell = cp.Constant(lT[max_iters-1])
        Q_T = cp.Constant(Qt[:, :, TT-1, max_iters-1])
        obj += cp.quad_form(delta_x[:, TT-1, max_iters-1], Q_T) + cp.matmul(gradT_ell, delta_x[:, TT-1, max_iters-1])

        # Define the constraints
        constraints = []
        for kk in range(max_iters-1):
            for tt in range(TT-1):
                grad1_f = cp.Constant(fx[:, :, tt, kk].T)
                grad2_f = cp.Constant(fu[:, :, tt, kk].T)
                constraints.append(delta_x[:, tt+1, kk] == cp.matmul(grad1_f, delta_x[:, tt, kk]) + cp.matmul(grad2_f, delta_u[:, tt, kk]))
            constraints.append(delta_x[:, 0, kk] == 0)

        # Define the problem
        problem = cp.Problem(cp.Minimize(obj), constraints)

        # Solve the problem
        problem.solve()

        # Get the optimal values
        delta_x_star = delta_x.value
        delta_u_star = delta_u.value
        print("delta_u and delta_x computed \n")

        #STEP 2: compute the input sequence
        print("Computing the input sequence...")
        
        stepsize=0.5
        for tt in range(TT-1):
            uu[:,tt,kk+1] = uu[:,tt,kk] + stepsize*delta_u_star[:,tt]
        print("Input sequence computed \n")

        #STEP 3: compute the state sequence
        print("Computing the state sequence...")
        
        for tt in range(TT-1):
            xx[:,tt+1,kk+1] = dyn.dynamics(xx[:,tt,kk],uu[:,tt,kk])[0]
        print("State sequence computed \n")

    xx_star = xx[:, :, max_iters - 1]
    uu_star = uu[:, :, max_iters - 1]
    uu_star[:, -1] = uu_star[:, -2]  # for plotting purposes       

    return xx_star, uu_star 