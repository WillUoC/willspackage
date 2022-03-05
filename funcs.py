import sys
from .exceptions import ArgumentError
import numpy as np
import deprecation

""" err(string).
    Prints 'string' and terminates program.
    30 Introduction to Python
"""
def err(string):
    print(string)
    input('Press return to exit')
    sys.exit()

""" a = solve_sonic_flow(theta, beta, mach)
    solve a third value given TWO
"""
def solve_sonic_flow(**kargs):
    from .search import newtonRaphson

    with np.errstate(divide='ignore'):
        if 'theta' not in kargs:
            beta = kargs['beta'] * np.pi/180
            mach = kargs['mach']
            gamma = 1.4

            sol = np.arctan(2 * (1/np.tan(beta)) * ((mach**2*np.sin(beta)**2 - 1)/(mach**2*(gamma+np.cos(2*beta))+2)))
            
            return(sol*180/np.pi)

        elif 'beta' not in kargs:
            theta = kargs['theta'] * np.pi/180
            mach = kargs['mach']
            gamma = 1.4
    #        function = lambda beta: 2 * mach**2 * np.sin(beta)**2 - 2 - np.tan(beta) * np.tan(theta) * (mach**2*(gamma+np.cos(2*beta)+2))
            function = lambda beta: 2 * (1/np.tan(beta)) * ((mach**2*np.sin(beta)**2 - 1)/(mach**2*(gamma+np.cos(2*beta))+2)) - np.tan(theta)
            function_derivative = derivative(function)

            sol = newtonRaphson(function, function_derivative, 0, np.pi/2)

            return(sol*180/np.pi)

        elif 'mach' not in kargs:
            theta = kargs['theta'] * np.pi/180
            beta = kargs['beta'] * np.pi/180
            gamma = 1.4

            function = lambda mach: 2 * (1/np.tan(beta)) * ((mach**2*np.sin(beta)**2 - 1)/(mach**2*(gamma+np.cos(2*beta))+2)) - np.tan(theta)
            function_derivative = derivative(function)
            sol = newtonRaphson(function, function_derivative, 0, 20)

            return(sol)

        else:
            raise ArgumentError("Too many variables defined")

@deprecation.deprecated(details="Function moved to approx module")
def derivative(f,method='central',h=1e-7):
    if method.lower() == 'central':
        return lambda x: (f(x+h) - f(x-h))/(2*h)
    if method.lower() == 'left':
        return lambda x: (f(x)-f(x-h))/h
    if method.lower() == 'right':
        return lambda x: (f(x + h) - f(x))/h
