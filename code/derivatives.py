from sympy import symbols, sin, cos, diff

# Definition of symbols to use sympy
x, y, psi, V, beta, psi_dot, delta, Fx, dt, mu, Fzf, Fzr, Fyf, Fyr, mass, Iz, a, b, g = symbols("x y psi V beta psi_dot delta Fx dt mu Fzf Fzr Fyf Fyr mass Iz a b g")

# Fzf = (mass * g * b) / (a + b)
# Fzr = (mass * g * a) / (a + b)

Fyf = mu * Fzf * (delta - (V * sin(beta) + a * psi_dot) / (V * cos(beta)))  # mu * Fzf * Bf
Fyr = mu * Fzr * (-(V * sin(beta) - b * psi_dot) / (V * cos(beta)))  # mu * Fzr * Br


def f_x():
    return x + dt * (V * cos(beta) * cos(psi) - V * sin(beta) * sin(psi))  # Discrete time


def f_y():
    return y + dt * (V * cos(beta) * sin(psi) + V * sin(beta) * cos(psi))  # Discrete time


def f_psi():
    return psi + dt * psi_dot  # Discrete time


def f_V():
    return V + dt * ((1 / mass) * (Fyr * sin(beta) + Fx * cos(beta - delta) + Fyf * sin(beta - delta)))  # Discrete time


def f_beta():
    return beta + dt * (1 / (mass * V) * (Fyr * cos(beta) + Fyf * cos(beta - delta) - Fx * sin(beta - delta)) - psi_dot)  # Discrete time


def f_psi_dot():
    return psi_dot + dt * (1 / Iz) * ((Fx * sin(delta) + Fyf * cos(delta)) * a - Fyr * b)  # Discrete time


if __name__ == "__main__":
    f_x_x = diff(f_x(), x)
    f_x_y = diff(f_x(), y)
    f_x_psi = diff(f_x(), psi)
    f_x_V = diff(f_x(), V)
    f_x_beta = diff(f_x(), beta)
    f_x_psi_dot = diff(f_x(), psi_dot)

    print(f"f_x_x = {f_x_x}")
    print(f"f_x_y = {f_x_y}")
    print(f"f_x_psi = {f_x_psi}")
    print(f"f_x_V = {f_x_V}")
    print(f"f_x_beta = {f_x_beta}")
    print(f"f_x_psi_dot = {f_x_psi_dot}")

    print("\n\n")

    f_y_x = diff(f_y(), x)
    f_y_y = diff(f_y(), y)
    f_y_psi = diff(f_y(), psi)
    f_y_V = diff(f_y(), V)
    f_y_beta = diff(f_y(), beta)
    f_y_psi_dot = diff(f_y(), psi_dot)

    print(f"f_y_x = {f_y_x}")
    print(f"f_y_y = {f_y_y}")
    print(f"f_y_psi = {f_y_psi}")
    print(f"f_y_V = {f_y_V}")
    print(f"f_y_beta = {f_y_beta}")
    print(f"f_y_psi_dot = {f_y_psi_dot}")

    print("\n\n")

    f_psi_x = diff(f_psi(), x)
    f_psi_y = diff(f_psi(), y)
    f_psi_psi = diff(f_psi(), psi)
    f_psi_V = diff(f_psi(), V)
    f_psi_beta = diff(f_psi(), beta)
    f_psi_psi_dot = diff(f_psi(), psi_dot)

    print(f"f_psi_x = {f_psi_x}")
    print(f"f_psi_y = {f_psi_y}")
    print(f"f_psi_psi = {f_psi_psi}")
    print(f"f_psi_V = {f_psi_V}")
    print(f"f_psi_beta = {f_psi_beta}")
    print(f"f_psi_psi_dot = {f_psi_psi_dot}")

    print("\n\n")

    f_V_x = diff(f_V(), x)
    f_V_y = diff(f_V(), y)
    f_V_psi = diff(f_V(), psi)
    f_V_V = diff(f_V(), V)
    f_V_beta = diff(f_V(), beta)
    f_V_psi_dot = diff(f_V(), psi_dot)

    print(f"f_V_x = {f_V_x}")
    print(f"f_V_y = {f_V_y}")
    print(f"f_V_psi = {f_V_psi}")
    print(f"f_V_V = {f_V_V}")
    print(f"f_V_beta = {f_V_beta}")
    print(f"f_V_psi_dot = {f_V_psi_dot}")

    print("\n\n")

    f_beta_x = diff(f_beta(), x)
    f_beta_y = diff(f_beta(), y)
    f_beta_psi = diff(f_beta(), psi)
    f_beta_V = diff(f_beta(), V)
    f_beta_beta = diff(f_beta(), beta)
    f_beta_psi_dot = diff(f_beta(), psi_dot)

    print(f"f_beta_x = {f_beta_x}")
    print(f"f_beta_y = {f_beta_y}")
    print(f"f_beta_psi = {f_beta_psi}")
    print(f"f_beta_V = {f_beta_V}")
    print(f"f_beta_beta = {f_beta_beta}")
    print(f"f_beta_psi_dot = {f_beta_psi_dot}")

    print("\n\n")

    f_psi_dot_x = diff(f_psi_dot(), x)
    f_psi_dot_y = diff(f_psi_dot(), y)
    f_psi_dot_psi = diff(f_psi_dot(), psi)
    f_psi_dot_V = diff(f_psi_dot(), V)
    f_psi_dot_beta = diff(f_psi_dot(), beta)
    f_psi_dot_psi_dot = diff(f_psi_dot(), psi_dot)

    print(f"f_psi_dot_x = {f_psi_dot_x}")
    print(f"f_psi_dot_y = {f_psi_dot_y}")
    print(f"f_psi_dot_psi = {f_psi_dot_psi}")
    print(f"f_psi_dot_V = {f_psi_dot_V}")
    print(f"f_psi_dot_beta = {f_psi_dot_beta}")
    print(f"f_psi_dot_psi_dot = {f_psi_dot_psi_dot}")

    print("\n\n")

    f_x_delta = diff(f_x(), delta)
    f_y_delta = diff(f_y(), delta)
    f_psi_delta = diff(f_psi(), delta)
    f_V_delta = diff(f_V(), delta)
    f_beta_delta = diff(f_beta(), delta)
    f_psi_dot_delta = diff(f_psi_dot(), delta)

    print(f"f_x_delta = {f_x_delta}")
    print(f"f_y_delta = {f_y_delta}")
    print(f"f_psi_delta = {f_psi_delta}")
    print(f"f_V_delta = {f_V_delta}")
    print(f"f_beta_delta = {f_beta_delta}")
    print(f"f_psi_dot_delta = {f_psi_dot_delta}")

    print("\n\n")

    f_x_Fx = diff(f_x(), Fx)
    f_y_Fx = diff(f_y(), Fx)
    f_psi_Fx = diff(f_psi(), Fx)
    f_V_Fx = diff(f_V(), Fx)
    f_beta_Fx = diff(f_beta(), Fx)
    f_psi_dot_Fx = diff(f_psi_dot(), Fx)

    print(f"f_x_Fx = {f_x_Fx}")
    print(f"f_y_Fx = {f_y_Fx}")
    print(f"f_psi_Fx = {f_psi_Fx}")
    print(f"f_V_Fx = {f_V_Fx}")
    print(f"f_beta_Fx = {f_beta_Fx}")
    print(f"f_psi_dot_Fx = {f_psi_dot_Fx}")
