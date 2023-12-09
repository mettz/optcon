import numpy as np
import dynamics as dyn
import cost as cst

def newton_method_optcon (xx_ref, uu_ref):
    #Step 0 (Initialization): consider an initla guess for the trajectory, so at iteration 0.
                            #It must contain all the trajectory (so until TT, time in main 10000), k  times (number of iterations)

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
    fx=np.zeros((ns,TT,max_iters))
    fu=np.zeros((ni,TT,max_iters))
    lT=np.zeros((TT,max_iters)) 
    
    fxx=np.zeros((ns, ns, TT, max_iters))
    fuu=np.zeros((ni, ni, TT, max_iters))
    fxu=np.zeros((ns, ni, TT, max_iters))
    
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

    delta_u = np.zeros((ni, TT, max_iters))
    delta_x = np.zeros((ns, TT, max_iters))

    for kk in range(max_iters):
        #Evaluate nabla1f, nabla2f, nabla1cost, nabla2cost, nablaTcost and hessians
        for tt in range(TT): #da 0 a 9999
            if tt == TT:
                lT[tt,kk]=cst.termcost(xx[:, tt, kk],xx_ref[:, tt, kk])[1] #VETTORE, per la costate equation
            else:
                at[:,tt,kk], bt[:,tt,kk] = cst.stagecost(xx[:, tt, kk], uu[:, tt, kk], xx_ref[:, tt], uu_ref[:, tt])[1:]
                fx[:,tt,kk], fu[:,tt,kk] = dyn.dynamics(xx[:, tt, kk], uu[:, tt, kk])[1:]

                fxx[:,:,tt,kk], fuu[:,:,tt,kk], fxu[:,:,tt,kk] = cst.hessian_cost()
            
        #Solve backward the costate equation 
        lmbd[:, TT, kk]=lT[:,kk] #Inizializzazione
        for tt in reversed(range(TT - 1)):  # integration backward in time

            At[tt,kk] = fx[tt,kk].T
            Bt[tt,kk] = fu[tt,kk].T

            lmbd[:, tt, kk] = lmbd[:, tt + 1, kk] @ At + at[tt,kk]

        #Compute for all t in TT-1, Qt,k ; Rt,k ; St,k, qt,k, rt,k ====> USE REGULARIZATION (slide 13)
        for tt in range(TT):
          if tt==TT:
            Qt[:,:,tt,kk] = fxx[:,:,tt,kk]
            qqt[:,tt,kk] = at[:,tt,kk]
          else:
            Qt[:,:,tt,kk] = fxx[tt,kk]
            Rt[:,:,tt,kk] = fuu[tt,kk]
            St[:,:,tt,kk] = fxu[tt,kk]
            qqt[:,tt,kk] = at[:,tt,kk]
            rrt[:,tt,kk] = bt[:,tt,kk]
             

        #Now compute the descent direction, solving the minimization problem to get delta_x and delta_u.
        #It's an affine LQ problem.
        
        PP[:,:,-1, kk] = Qt[:,:,TT-1, kk]
        pp[:,-1,kk] = qqt[:,TT-1,kk]

        for tt in reversed(range(TT-1)):
          MMt_inv[:,:,tt,kk] = np.linalg.inv(Rt[:,:,tt,kk] + Bt[:,:,tt,kk].T @ PP[:,:,tt-1,kk] @ Bt[:,:,tt,kk])
          mmt[:,tt,kk] = rrt[:,tt,kk] + Bt[:,:,tt,kk].T @ pptp
          
          PP[:,:,tt,kk] = At[:,:,tt,kk].T @ PP[:,:,tt-1,kk] @ At[:,:,tt,kk] - (Bt[:,:,tt,kk].T@PP[:,:,tt-1,kk]@At[:,:,tt,kk] + St[:,:,tt,kk]).T @ MMt_inv[:,:,tt,kk] @ (Bt[:,:,tt,kk].T@PP[:,:,tt-1,kk]@At[:,:,tt,kk] + St[:,:,tt,kk]) + Qt[:,:,tt,kk]
          pp[:,tt,kk] = At[:,:,tt,kk].T @ pp[:,tt-1,kk] - (Bt[:,:,tt,kk].T@PP[:,:,tt-1,kk]@At[:,:,tt,kk] + St[:,:,tt,kk]).T @ MMt_inv[:,:,tt,kk] @ mmt[:,tt,kk] + qqt[:,tt,kk]

 

        # Evaluate KK
          for tt in range(TT-1):

            PPtp = PP[:,:,tt+1]
            pptp = pp[:,tt+1][:,None]

            # Check positive definiteness

            MMt_inv = np.linalg.inv(Rt[:,:,tt,kk] + Bt[:,:,tt,kk].T @ PP[:,:,tt+1,kk] @ Bt[:,:,tt,kk])
            mmt = rrt[:,tt,kk] + Bt[:,:,tt,kk].T @ pp[:,tt+1,kk]

            # for other purposes we could add a regularization step here...

            KK[:,:,tt] = -MMt_inv@(Bt[:,:,tt,kk].T@PP[:,:,tt+1,kk]@At[:,:,tt,kk] + St[:,:,tt,kk])
            sigma_t[:,tt,kk] = -MMt_inv@mmt


          # Evaluate delta_u
          for tt in range(TT - 1):
            delta_u[:, tt] = KK[:,:,tt]@xx[:, tt] + sigma_t[:,tt,kk]
            delta_x[:,tt+1] = At[:,:,tt,kk]@xx[:,tt] + Bt[:,:,tt,kk]@uu[:, tt, kk]

          #STEP 2: compute the input sequence
          stepsize=0.5
          for tt in range(TT-1):
            uu[:,tt,kk+1] = uu[:,tt,kk] + stepsize*delta_u[:,tt]

          #STEP 3: compute the state sequence
          for tt in range(TT-1):
            xx[:,tt+1,kk+1] = dyn.dynamics(xx[:,tt,kk],uu[:,tt,kk])[0]