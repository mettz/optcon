import numpy as np

def newton_method(xx, grad, tolerance, max_iter):
    xx_old = xx
    for i in range(max_iter):
        xx = xx_old - np.linalg.solve(grad,xx_old)
        if np.linalg.norm(xx - xx_old) < tolerance:
            break
        xx_old = xx

    print(f"Newton method converged in {i} iterations")
    return xx