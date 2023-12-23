# discretization step
DT = 0.01

# final time of the trajectory
TF = 5

# number of time steps
TT = int(TF / DT)

STATES = ["V", "beta", "psi_dot"]
INPUTS = ["delta", "Fx"]

NUMBER_OF_STATES = len(STATES)
NUMBER_OF_INPUTS = len(INPUTS)
