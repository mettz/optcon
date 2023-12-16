import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from newton_method_optcon_cvxpy import TT, ni, ns

xx_ref = np.zeros((ns, TT))
uu_ref = np.zeros((ni, TT))
  
def step_trajectory(initial_eq_state, initial_eq_input, final_eq_state, final_eq_input):
   # Initialization of the reference curve
    xx_ref = np.zeros((ns, TT))
    uu_ref = np.zeros((ni, TT))

    # Definition of the reference curve
    for i in range(TT):
        if i < (TT / 2):
            xx_ref[:, i] = initial_eq_state
            uu_ref[:, i] = initial_eq_input
        else:
            xx_ref[:, i] = final_eq_state
            uu_ref[:, i] = final_eq_input 
    return xx_ref, uu_ref

def smooth_trajectory(initial_eq_state, initial_eq_input, final_eq_state, final_eq_input):

    final_time=3
    x = np.array([0, final_time])
    for i in range(ns):
        
        # Given points
  
        y = initial_eq_state[i], final_eq_state[i]

        # Interpolate using CubicSpline
        interp = CubicSpline(x, y, bc_type='clamped')

        # Generate points for the interpolated polynomial
        x_interp = np.linspace(0, final_time, 1000)
        y_interp = interp(x_interp)
     
        xx_ref_1 = np.tile(initial_eq_state[i], (TT//2)-(len(y_interp)//2))
        xx_ref_2 = np.tile(final_eq_state[i], (TT//2)-(len(y_interp)//2))
        xx_ref[i,:] = np.concatenate((xx_ref_1,y_interp, xx_ref_2), axis=0)


    for i in range(ni):
        
        # Given points
        y = initial_eq_input[i], final_eq_input[i]

        # Interpolate using CubicSpline
        interp = CubicSpline(x, y, bc_type='clamped')

        # Generate points for the interpolated polynomial
        x_interp = np.linspace(0, final_time, 1000)
        y_interp = interp(x_interp)
     
        uu_ref_1 = np.tile(initial_eq_input[i], (TT//2)-(len(y_interp)//2))
        uu_ref_2 = np.tile(final_eq_input[i], (TT//2)-(len(y_interp)//2))
        uu_ref[i,:] = np.concatenate((uu_ref_1,y_interp, uu_ref_2), axis=0)

    return xx_ref, uu_ref