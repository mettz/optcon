import dynamics as dyn
import numpy as np
from newton_method import newton_method
import sys

def main():
    # write output to file
    sys.stdout = open('output.txt', 'w')
    for _ in range(0, 10):
        x_init = np.random.rand(6)
        u_init = np.random.rand(2)
        print(f"x_init: {x_init}")
        print(f"u_init: {u_init}")
        xx, fx, fu = dyn.dynamics(x_init, u_init)
        tolerance = 1e-6
        max_iter = 1000

        equilibrium = newton_method(x_init, fx, tolerance, max_iter)
        print(f"Equilibrium point: {equilibrium}")


if __name__ == "__main__":
    main()


#Equilibrium point: [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  4.99682041e-03 1.23996445e-01 -1.09429904e-07]
#x_init: [0.33995582 0.12381764 0.77293919 0.00728807 0.29958201 0.00707914]
#u_init: [0.18109002 0.36867134]

#Equilibrium point: [0.00000000e+00 0.00000000e+00 0.00000000e+00 3.55515944e-07 4.78161346e-01 1.16494770e-08]
#x_init: [0.30977437 0.59922563 0.34674899 0.09932157 0.46665249 0.89194707]
#u_init: [0.97547049 0.96346462]


# Equilibrium point: [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  8.03380024e-07  1.16747689e-01 -1.44442985e-07]
# x_init: [0.81997678 0.39564295 0.97369565 0.75586466 0.40201241 0.27849996]
# u_init: [0.22804877 0.28188647]