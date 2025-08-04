import argparse
import numpy as np
from methods import sgd, cg, lbfgs
from functions import get_function

# python3 main.py --method sgd --func rosenbrock --lr 0.001 --max-iter 1000
# python3 main.py --method cg --func quadratic --max-iter 1000 --plot True
# python3 main.py --method cg --func rosenbrock --max-iter 1000 --init 0.5 0.5 --plot True

def main():
    parser = argparse.ArgumentParser(
        prog = "Optimization",
        description = "Runs provided optimization method")
    parser.add_argument("--method", type=str, choices=["sgd", "cg", "lbfgs"], required=True)
    parser.add_argument("--func", type=str, choices=["rosenbrock", "rastrigin", "quadratic"], required=True)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--init", type=float, nargs="+", default=[-1.2, 1.0])
    parser.add_argument("--plot", type=bool, default=False)
    parser.add_argument("--tol", type=float, default=1e6)

    args = parser.parse_args()

    f, grad_f = get_function(args.func)
    x0 = args.init

    if args.method == "sgd":
        result = sgd.run(f, grad_f, x0, lr=args.lr, max_iter=args.max_iter)
    elif args.method == "cg":
        result = cg.run(f, grad_f, x0, plot=args.plot, max_iter=args.max_iter, func_name=args.func)
    elif args.method == "lbfgs":
        result = lbfgs.run(f, grad_f, x0, max_iter=args.max_iter)
        

 #   print(f"Result: {result}")

if __name__ == "__main__":
    main()