# optimization-methods-comparison
This repo compares gradient descent (GD), conjugate gradient (CG), and BFGS methods on optimizing several different functions.  

## Usage
Run with `python --method {sgd,cg,bfgs} 
            --func {rosenbrock,rastrigin,quadratic}
            --lr {FLOAT}
            --max-iter {FLOAT}
            --init {FLOAT} {FLOAT}
            --tol {FLOAT}
            --plot {BOOLEAN}
            --steepness {BOOLEAN}`
            
or `python3 main.py -h` to get instructions

> Parameters

**method**&ensp;:&ensp;*{'sgd', 'cg', 'bfgs'}*

&emsp;The optimization method to use

**function**&ensp;:&ensp;*{'rosenbrock', 'rastrigin', 'quadratic'}*

&emsp;The function to optimize

**lr**&ensp;:&ensp;*float, optional*

&emsp;The learning rate. Only necessary if using sgd

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