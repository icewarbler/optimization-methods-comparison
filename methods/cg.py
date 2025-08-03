import numpy as np

def golden_section_search(fun, a, b):
    while abs(a-b) > tol:
        tau = (np.sqrt(5) - 1) / 2
        m1 = a + (1-tau)*(b-a)
        m2 = a + tau*(b-a)

        if fun(m1) > fun(m2):
            a = m1
        else:
            b = m2

    return a
    
def run(f, grad_f, x0, max_iter=1000):
    return 0