import numpy as np

import dynamics as dyn
import constants



def LQR_tracking(xx_ref, uu_ref):

    x0 = xx_ref[:,0]-np.array([0.5,0.5,0.5])

    #Step 0: define the cost matrices
    QQ_reg = np.diag([25, 30, 35])
    QQ_reg_T = QQ_reg
    RR_reg = np.diag([30, 20])

    #Step 1: linearize the dynamics around the reference trajectory (slide 3 pacco 9)
    AA = np.zeros((constants.NUMBER_OF_STATES, constants.NUMBER_OF_STATES, constants.TT-1))
    BB = np.zeros((constants.NUMBER_OF_STATES, constants.NUMBER_OF_INPUTS, constants.TT-1))
    for tt in range(constants.TT-1):

        AA[:,:,tt] = dyn.dynamics(xx_ref[:, tt], uu_ref[:, tt])[1].T
        BB[:,:,tt] = dyn.dynamics(xx_ref[:, tt], uu_ref[:, tt])[2].T

    PP = np.zeros((constants.NUMBER_OF_STATES,constants.NUMBER_OF_STATES,constants.TT))
    KK = np.zeros((constants.NUMBER_OF_INPUTS,constants.NUMBER_OF_STATES,constants.TT))
    
    PP[:,:,-1] = QQ_reg_T
    
    # Solve Riccati equation
    for tt in reversed(range(constants.TT-1)):
        QQt = QQ_reg
        RRt = RR_reg
        AAt = AA[:,:,tt]
        BBt = BB[:,:,tt]
        PPtp = PP[:,:,tt+1]
        
        PP[:,:,tt] = QQt + AAt.T@PPtp@AAt - \
            + (AAt.T@PPtp@BBt)@np.linalg.inv((RRt + BBt.T@PPtp@BBt))@(BBt.T@PPtp@AAt)
    
    # Evaluate KK
    
    for tt in range(constants.TT-1):
        QQt = QQ_reg
        RRt = RR_reg
        AAt = AA[:,:,tt]
        BBt = BB[:,:,tt]
        PPtp = PP[:,:,tt+1]
        
        KK[:,:,tt] = -np.linalg.inv(RRt + BBt.T@PPtp@BBt)@(BBt.T@PPtp@AAt)

    xx = np.zeros((constants.NUMBER_OF_STATES,constants.TT))
    uu = np.zeros((constants.NUMBER_OF_INPUTS,constants.TT))

    xx[:,0] = x0

    for tt in range(constants.TT-1):
        uu[:,tt] = uu_ref[:,tt] + KK[:,:,tt]@(xx[:,tt] - xx_ref[:,tt])
        xx[:,tt+1] = dyn.dynamics(xx[:,tt],uu[:,tt])[0]

    return xx, uu