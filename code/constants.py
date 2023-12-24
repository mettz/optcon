# Discretization step
DT = 0.01

# Final time of the trajectory
TF = 5

# Number of time steps
TT = int(TF / DT)

# State and input variables
STATES = ["V", "beta", "psi_dot"]
INPUTS = ["delta", "Fx"]

# Number of states and inputs
NUMBER_OF_STATES = len(STATES)
NUMBER_OF_INPUTS = len(INPUTS)
