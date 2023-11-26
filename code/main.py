import dynamics as dyn
import numpy as np
from newton_method import newton_method


def main():
    x_init = np.array([1, 1, 1, 1, 1, 1])
    u_init = np.array([0, 0])
    xx, fx, fu = dyn.dynamics(x_init, u_init)
    tolerance = 1e-6
    max_iter = 1000

    equilibrium = newton_method(x_init, fx, tolerance, max_iter)
    print(f"Equilibrium point: {equilibrium}")


if __name__ == "__main__":
    main()
