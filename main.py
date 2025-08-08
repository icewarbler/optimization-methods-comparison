import argparse
import numpy as np
from methods import nm, cg, bfgs
from functions import get_function
import logging
logger = logging.getLogger(__name__)

def main():

    parser = argparse.ArgumentParser(
        prog = "Optimization",
        description = "Runs specified optimization method")
    parser.add_argument("--method", type=str, choices=["nm", "cg", "bfgs"], required=True)
    parser.add_argument("--func", type=str, choices=["rosenbrock", "rastrigin", "quadratic"], required=True)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--init", type=float, nargs="+", default=[-1.2, 1.0])
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--plot", type=bool, default=False)
    parser.add_argument("--steepness", type=bool, default=False)

    args = parser.parse_args()

    f, grad_f = get_function(args.func)
    x0 = np.array(args.init)

    if args.method == "nm":
        point, value, iters = nm.run(f, x0, max_iter=args.max_iter, tol=args.tol, plot=args.plot, func_name=args.func, steepness=args.steepness)
    elif args.method == "cg":
        point, value, iters = cg.run(f, grad_f, x0, tol=args.tol, plot=args.plot, max_iter=args.max_iter, func_name=args.func, steepness=args.steepness)
    elif args.method == "bfgs":
        point, value, iters = bfgs.run(f, grad_f, x0, tol=args.tol, max_iter=args.max_iter, func_name=args.func, plot=args.plot, steepness=args.steepness)
        

    print(f"At point ({point[0]:.4f}, {point[1]:.4f}) with value {value:.4f} in {iters} iterations")

if __name__ == "__main__":
    main()