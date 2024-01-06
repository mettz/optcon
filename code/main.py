import signal
import argparse
import numpy as np

import constants
import curves
import equilibrium as eq
import solvers
import plots
import trackers

signal.signal(signal.SIGINT, signal.SIG_DFL)


def main(args):
    plotter = plots.Plotter(**vars(args))
    plotter.derivatives_plot()

    V_des = [1, 1]
    psi_dot_des = [0, 0.1]

    eq1 = eq.find(V=V_des[0], psi_dot=psi_dot_des[0])
    print(eq1)
    eq2 = eq.find(V=V_des[1], psi_dot=psi_dot_des[1])
    print(eq2)

    xx_eq1 = np.array([V_des[0], eq1.beta, psi_dot_des[0]])
    uu_eq1 = np.array([eq1.delta, eq1.Fx])
    plotter.verify_equilibria(eq1)

    xx_eq2 = np.array([V_des[1], eq2.beta, psi_dot_des[1]])
    uu_eq2 = np.array([eq2.delta, eq2.Fx])
    plotter.verify_equilibria(eq2)

    curve = None
    if args.ref_curve == "step":
        curve = curves.step
    elif args.ref_curve == "cubic":
        curve = curves.cubic_spline
    else:
        raise ValueError(f"Invalid reference curve {curve}")

    xx_ref = curve(start=xx_eq1, end=xx_eq2, steps=constants.TT)
    uu_ref = curve(start=uu_eq1, end=uu_eq2, steps=constants.TT)

    plotter.reference_curves(xx_ref, uu_ref)

    xx_star = None
    uu_star = None

    if args.solver == "gradient":
        xx_star, uu_star = solvers.gradient(xx_ref, uu_ref, plotter)
    elif args.solver == "newton":
        xx_star, uu_star = solvers.newton(xx_ref, uu_ref, plotter)
    else:
        raise ValueError(f"Invalid solver {args.solver}")

    plotter.following_plots(xx_ref, uu_ref, xx_star, uu_star)

    # Defining percentage of errors in state and input
    error = []
    for i in range(constants.NUMBER_OF_STATES):
        error.append(np.abs(xx_ref[i, :] - xx_star[i, :]))
        print(f"Error in state {constants.STATES[i]}: {np.mean(error)}")

    for i in range(constants.NUMBER_OF_INPUTS):
        error.append(np.abs(uu_ref[i, :] - uu_star[i, :]))
        print(f"Error in input {constants.INPUTS[i]}: {np.mean(error)}")

    # Defining overshooting in input
    overshooting = []
    for i in range(constants.NUMBER_OF_INPUTS):
        max_input_star = np.max(uu_star[i, :])
        max_input_ref = np.max(uu_ref[i, :])
        overshooting.append((max_input_star - max_input_ref) / max_input_ref)
        print(f"Overshooting in input {constants.INPUTS[i]}: {overshooting[i]}")

    print("Starting LQR tracking...")
    xx_lqr, uu_lqr = trackers.lqr(xx_star, uu_star)
    plotter.lqr_plots(xx_star, uu_star, xx_lqr, uu_lqr)

    print("Starting MPC tracking...")
    xx_mpc, uu_mpc = trackers.mpc(xx_star, uu_star)
    plotter.mpc_plots(xx_star, uu_star, xx_mpc, uu_mpc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous Car Optimization")

    parser.add_argument("--show-derivative-plots", action="store_true", default=False, help="Show derivatives verification plots")

    parser.add_argument("-c", "--ref-curve", type=str, choices=["step", "cubic"], default="step", help="Reference curve to follow")

    parser.add_argument("-s", "--solver", type=str, choices=["gradient", "newton"], default="newton", help="Solver to use")

    parser.add_argument("--show-ref-curves-plots", action="store_true", default=False, help="Show reference curves plots")

    parser.add_argument("--show-verify-equilibria", action="store_true", default=False, help="Show verification of equilibria plots")

    parser.add_argument("--show-solver-plots", action="store_true", default=False, help="Show chosen solver plots such as armijo, descent, costs etc...")

    parser.add_argument("--show-following-plots", action="store_true", default=False, help="Show optimal trajectories plots")

    parser.add_argument("--show-lqr-plots", action="store_true", default=False, help="Show LQR tracking plots")

    parser.add_argument("--show-mpc-plots", action="store_true", default=False, help="Show MPC tracking plots")

    parser.add_argument("-q", "--quiet", action="store_true", default=False, help="Do not show any plots (except the ones specified with --show-* flags)")

    main(parser.parse_args())
