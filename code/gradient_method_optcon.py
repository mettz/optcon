import numpy as np
import matplotlib.pyplot as plt
import dynamics as dyn
import cost as cst

import plots

import signal

signal.signal(signal.SIGINT, signal.SIG_DFL)

plt.rcParams["figure.figsize"] = (10, 8)
plt.rcParams.update({"font.size": 22})

# Algorithm parameters
#max_iters = int(3e2)
max_iters = 10
stepsize_0 = 0.001

# Armijo parametrs
cc = 0.5
beta = 0.7
armijo_maxiters = 20 

term_cond = 1e-3

visu_armijo = True

# Trajectory parameters
tf = 10  # final time in seconds

dt = dyn.dt  # get discretization step from dynamics
ns = dyn.number_of_states
ni = dyn.number_of_inputs

TT = int(tf / dt)  # discrete-time samples
print("TT", TT)

# Initial guess
xx_init = np.ones((ns, TT)) # 3x10000
# xx_init contiene tutta la guess trajectory

print("xx_init", xx_init)
uu_init = np.ones((ni, TT)) # 2x10000

#xx_init[:,0] = np.array()
######################################
# Arrays to store data
######################################

xx = np.zeros((ns, TT, max_iters))  # state seq.
uu = np.zeros((ni, TT, max_iters))  # input seq.

lmbd = np.zeros((ns, TT, max_iters))  # lambdas - costate seq.

deltau = np.zeros((ni, TT, max_iters))  # Du - descent direction
dJ = np.zeros((ni, TT, max_iters))  # DJ - gradient of J wrt u

JJ = np.zeros(max_iters)  # collect cost
descent = np.zeros(max_iters)  # collect descent direction
descent_arm = np.zeros(max_iters)  # collect descent direction


def gradient_method(xx_ref, uu_ref):
    print("Starting the computation of the optimal trajectory...")

    kk = 0
    iters = max_iters # 300

    #xx[:, :, 0] = xx_init 
    #print("xx_ref[:,0]", xx_ref[:,0])
    xx[:, :, 0] = xx_ref[:,0,None]
    #uu[:, :, 0] = uu_init
    #uu_init = uu_ref[:,0, None]
    uu[:, :, 0] = uu_ref[:,0, None]

    x0 = xx_ref[:, 0]

    for kk in range(iters - 1): #da 0 a 299
        JJ[kk] = 0
        # calculate cost
        for tt in range(TT - 1): #da 0 a 9999
            temp_cost = cst.stagecost(xx[:, tt, kk], uu[:, tt, kk], xx_ref[:, tt], uu_ref[:, tt])[0]
            JJ[kk] += temp_cost

        temp_cost = cst.termcost(xx[:, -1, kk], xx_ref[:, -1])[0]
        JJ[kk] += temp_cost
        
        # Descent direction calculation

        lmbd_temp = cst.termcost(xx[:, TT - 1, kk], xx_ref[:, TT - 1])[1]
        lmbd[:, TT - 1, kk] = lmbd_temp.squeeze()

        for tt in reversed(range(TT - 1)):  # integration backward in time
            at, bt = cst.stagecost(xx[:, tt, kk], uu[:, tt, kk], xx_ref[:, tt], uu_ref[:, tt])[1:]
            fx, fu = dyn.dynamics(xx[:, tt, kk], uu[:, tt, kk])[1:]

            At = fx.T
            Bt = fu.T
            
            lmbd_temp = At.T @ lmbd[:, tt + 1, kk][:, None] + at  # costate equation
            dJ_temp = Bt.T @ lmbd[:, tt + 1, kk][:, None] + bt  # gradient of J wrt u
            deltau_temp = -dJ_temp

            lmbd[:, tt, kk] = lmbd_temp.squeeze()
            dJ[:, tt, kk] = dJ_temp.squeeze()
            deltau[:, tt, kk] = deltau_temp.squeeze()

            descent[kk] += deltau[:, tt, kk].T @ deltau[:, tt, kk]
            descent_arm[kk] += dJ[:, tt, kk].T @ deltau[:, tt, kk] #

        ##################################
        # Stepsize selection - ARMIJO
        ##################################

        stepsizes = []  # list of stepsizes
        costs_armijo = [] 

        stepsize = stepsize_0

        for ii in range(armijo_maxiters):
            # temp solution update

            xx_temp = np.zeros((ns, TT))
            uu_temp = np.zeros((ni, TT))

            xx_temp[:, 0] = x0

            for tt in range(TT - 1):
                uu_temp[:, tt] = uu[:, tt, kk] + stepsize * deltau[:, tt, kk]
                xx_temp[:, tt + 1] = dyn.dynamics(xx_temp[:, tt], uu_temp[:, tt])[0]

            # temp cost calculation
            JJ_temp = 0

            for tt in range(TT - 1):
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
                if visu_armijo and kk % 10 == 0:
                    plots.armijo_plot(stepsize_0, stepsizes, costs_armijo, descent_arm, JJ, kk, cc, ns, ni, TT, x0, uu, deltau, dyn, cst, xx_ref, uu_ref)
                    print("Armijo stepsize = {:.3e}".format(stepsize))
                    break

       
        ############################
        # Update the current solution
        ############################
        
        # stepsize = 0.01 #Constant stepsize

        xx_temp = np.zeros((ns, TT))
        uu_temp = np.zeros((ni, TT))

        xx_temp[:, 0] = x0

        for tt in range(TT - 1):
            uu_temp[:, tt] = uu[:, tt, kk] + stepsize * deltau[:, tt, kk]
            xx_temp[:, tt + 1] = dyn.dynamics(xx_temp[:, tt], uu_temp[:, tt])[0]

        xx[:, :, kk + 1] = xx_temp
        uu[:, :, kk + 1] = uu_temp

        ############################
        # Termination condition
        ############################

        print("Iter = {}\t Descent = {:.3e}\t Cost = {:.3e}".format(kk, descent[kk], JJ[kk]))

        if descent[kk] <= term_cond:
            iters = kk

            break

    xx_star = xx[:, :, iters - 1]
    uu_star = uu[:, :, iters - 1]
    uu_star[:, -1] = uu_star[:, -2]  # for plotting purposes

    return xx_star, uu_star
