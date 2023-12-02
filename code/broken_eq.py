import numpy as np
from equilibrium import find_equilibrium_point, nonlinear_system_discretized, nonlinear_system_continuous
from plots import plot_equilibria, plot_with_equilibria

if __name__ == "__main__":
    #PUNTO DI EQUILIBRIO ROTTO:
    beta_des_1 = 20.0
    V_des_1 = 1.0
    eq1 = find_equilibrium_point(nonlinear_system_discretized, initial_guess=[0, 0, 0], args = (V_des_1, beta_des_1))
    print("1st eq point: ", eq1)
    print("Diff: ", np.isclose(nonlinear_system_discretized(eq1, V_des_1, beta_des_1), np.zeros(3)))
    # Valore non corrispondente di psi_dot nel grafico ottenuto: non va bene!
    # Per il resto sembra perfettamente un punto di equilibrio
    eq1_c = find_equilibrium_point(nonlinear_system_continuous, initial_guess=[0, 0, 0], args=(V_des_1, beta_des_1))
    print("1st eq point: ", eq1_c)
    print("Diff: ", np.isclose(nonlinear_system_continuous(eq1_c, V_des_1, beta_des_1), np.zeros(3)))
    plot_equilibria(eq1, beta_des_1, V_des_1)
    plot_equilibria(eq1_c, beta_des_1, V_des_1) # using continuous dynamics
    plot_with_equilibria(eq1, V_des_1, beta_des_1)