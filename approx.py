"""Approximation Package
    by: William Black

    Various functions for approximating mathematical expressions
"""

def derivative_first_order(f,method='forward',h=0.01):
    if method.lower() == 'backward':
        return lambda x: (f(x)-f(x-h))/h
    if method.lower() == 'forward':
        return lambda x: (f(x + h) - f(x))/h

def double_derivative_first_order(f, method='forward', h=0.01):
    if method.lower() == 'backward':
        return lambda x: (f(x)-2*f(x-h)+f(x-2*h))/(h**2)
    if method.lower() == 'forward':
        return lambda x: (f(x) - 2*f(x+h) + f(x+2*h))/(h**2)

def derivative_second_order(f, method='central', h=0.01):
    if method.lower() == 'central':
        return lambda x: (f(x+h) - f(x-h))/(2*h)
    if method.lower() == 'backward':
        return lambda x: (3*f(x)-4*f(x-h)+f(x-2*h))/(2*h)
    if method.lower() == 'forward':
        return lambda x: (4*f(x + h) - 3*f(x) - f(x + 2*h))/(2*h)

def double_derivative_second_order(f, method='central', h=0.01):
    if method.lower() == 'central':
        return lambda x: (f(x+h) - 2*f(x) + f(x-h))/(h**2)
    if method.lower() == 'backward':
        return lambda x: (-1*f(x-3*h) + 4*f(x-2*h) - 5*f(x-h) + 2*f(x))/(h**2)
    if method.lower() == 'forward':
        return lambda x: (2*f(x) - 5*f(x + h) + 4*f(x + 2*h) - f(x + 3*h))/(h**2)