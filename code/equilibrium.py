import numpy as np
from scipy.optimize import fsolve
import textwrap

import constants
import dynamics as dyn


def find(*, V, beta):
    eq = Equilibrium(V, beta)
    eq.compute()
    return eq


class Equilibrium:
    def __init__(self, V, beta):
        self.V = V
        self.beta = beta
        self.psi_dot = None
        self.delta = None
        self.Fx = None
        self.is_close = False

    # Method that finds the equilibrium point of the system for a given V and beta pair
    # It always starts from an initial guess = [0, 0, 0]
    def compute(self):
        initial_guess = np.zeros(constants.NUMBER_OF_STATES)

        # Definition of the non linear system to solve which is the discretized version of the dynamics
        def f(var, V, beta):
            psi_dot, delta, Fx = var

            xx = np.array([V, beta, psi_dot])
            uu = np.array([delta, Fx])

            eq1 = constants.DT * dyn.V(xx, uu)
            eq2 = constants.DT * dyn.beta(xx, uu)
            eq3 = constants.DT * dyn.psi_dot(xx, uu)

            return [eq1, eq2, eq3]

        # Find the equilibrium point
        eq = fsolve(f, initial_guess, args=(self.V, self.beta))
        self.psi_dot, self.delta, self.Fx = eq

        # Mathemathical check of the equilibrium point
        self.is_close = np.isclose(f(eq, self.V, self.beta), np.zeros(constants.NUMBER_OF_STATES)).all()

    def __str__(self):
        return textwrap.dedent(
            f"""
            Equilibrium point for (V={self.V}, beta={self.beta}):
              - psi_dot = {self.psi_dot}
              - delta = {self.delta}
              - Fx = {self.Fx}
            Is close? {'Yes' if self.is_close else 'No'}
            """
        )
