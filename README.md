# optimization-methods-comparison
This repo compares Nelder-Mead (NM), conjugate gradient (CG), and BFGS methods on optimizing several different functions.  

## Usage
Run with `python --method {nm,cg,bfgs} 
            --func {rosenbrock,rastrigin,quadratic}
            --max-iter {FLOAT}
            --init {FLOAT} {FLOAT}
            --tol {FLOAT}
            --plot {BOOLEAN}
            --steepness {BOOLEAN}`
            
or `python3 main.py -h` to get instructions

> Parameters

**method**&ensp;:&ensp;*{'nm', 'cg', 'bfgs'}*

&emsp;The optimization method to use

**function**&ensp;:&ensp;*{'rosenbrock', 'rastrigin', 'quadratic'}*

&emsp;The function to optimize

**max_iter**&ensp;:&ensp;*float, optional*

&emsp;The maximum number of iterations for the method

**init**&ensp;:&ensp;*float, optional*

&emsp;The inital values. It should be structured without paranthesis/brackets; i.e. `2 2`

**tol**&ensp;:&ensp;*float, optional*

&emsp;The tolerance at which the iterations will stop

**plot**&ensp;:&ensp;*bool, optional*

&emsp;Whether or not the function will be plotted. Defaults to `False`

**steepness**&ensp;:&ensp;*bool, optional*

&emsp;Whether or not the method's steepness will be plotted. Defaults to `False`

## Usage Notes

- Rastrigin is not optimal for the conjugate gradient method, as there are repeated minima

- Rosenbrock is ill-conditioned, so expect a very slow convergence when using conjugate gradient. Expect to require a very high tolerance (tol≈9e-02), and, even then, expect to see oscillatory convergence
