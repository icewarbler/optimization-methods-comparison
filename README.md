# optimization-methods-comparison

This repository compares three optimization methods — **Nelder-Mead (NM)**, **Conjugate Gradient (CG)**, and **BFGS** — on several benchmark functions.

## Overview

The project allows you to test and visualize different optimization methods on common mathematical functions:

- **Algorithms**:  
  - Nelder-Mead (NM)  
  - Conjugate Gradient (CG)  
  - BFGS  

- **Benchmark functions**:  
  - Rosenbrock  
  - Rastrigin  
  - Quadratic 

## Usage


Run the script with:

```bash
python main.py --method {nm,cg,bfgs} \
               --func {rosenbrock,rastrigin,quadratic} \
               --max-iter {INT} \
               --init {FLOAT} {FLOAT} \
               --tol {FLOAT} \
               --plot {BOOLEAN} \
               --steepness {BOOLEAN}
```

or check the help message:

 `python3 main.py -h`

 ### Parameters

| Parameter      | Type                  | Description |
|----------------|---------------------|-------------|
| `--method`     | str                   | Optimization method to use. Choices: `nm`, `cg`, `bfgs` |
| `--func`       | str                   | Function to optimize. Choices: `rosenbrock`, `rastrigin`, `quadratic` |
| `--max-iter`   | int, optional         | Maximum number of iterations (default depends on method) |
| `--init`       | float float, optional | Initial values for the optimizer, e.g., `2 2` |
| `--tol`        | float, optional       | Stopping tolerance for the method |
| `--plot`       | bool, optional        | Whether to plot the function (default: `False`) |
| `--steepness`  | bool, optional        | Whether to plot the method's steepness (default: `False`) |


## Usage Notes

- **Rastrigin** is not optimal for the conjugate gradient method, as there are repeated minima

- **Rosenbrock** is ill-conditioned, so expect a very slow convergence when using conjugate gradient. Expect to require a very high tolerance (`tol≈9e-02`), and, even then, expect to see oscillatory convergence
