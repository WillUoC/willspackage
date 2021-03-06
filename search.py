from .linalg import err, gaussPivot as error, gaussPivot
import math
import numpy as np
from .exceptions import RootError
import cmath
from random import random


''' x1,x2 = rootsearch(f,a,b,dx).
    Searches the interval (a,b) in increments dx for
    the bounds (x1,x2) of the smallest root of f(x).
    Returns x1 = x2 = None if no roots were detected.
'''
def rootsearch(f,a,b,dx):
    x1 = a; f1 = f(a)
    x2 = a + dx; f2 = f(x2)

    while np.sign(f1) == np.sign(f2):

        if x1 >= b: return None,None
        x1 = x2
        f1 = f2
        x2 = x1 + dx
        f2 = f(x2)

    else:
        return x1,x2


''' root = bisection(f,x1,x2,switch=0,tol=1.0e-9).
    Finds a root of f(x) = 0 by bisection.
    The root must be bracketed in (x1,x2).
    Setting switch = 1 returns root = None if
    f(x) increases upon bisection.
'''
def bisect(f,x1,x2,switch=1,tol=1.0e-9):
    f1 = f(x1)

    if f1 == 0.0: return x1

    f2 = f(x2)

    if f2 == 0.0: return x2

    if np.sign(f1) == np.sign(f2):
        RootError("Root not bracketed")

    n = int(math.ceil(math.log(abs(x2 - x1)/tol)/math.log(2.0)))
    
    for i in range(n):
        x3 = 0.5*(x1 + x2); f3 = f(x3)

        if (switch == 1) and (abs(f3) > abs(f1)) and (abs(f3) > abs(f2)):
            return None
    
        if f3 == 0.0: return x3

        if np.sign(f2)!= np.sign(f3): 
            x1 = x3
            f1 = f3
        else:
            x2 = x3
            f2 = f3

    return (x1 + x2)/2.0


def eval_poly_derivatives(a,x):
    n = len(a) - 1
    p = a[n]
    dp = 0.0 + 0.0j
    ddp = 0.0 + 0.0j
    for i in range(1,n+1):
        ddp = ddp*x + 2.0*dp
        dp = dp*x + p
        p = p*x + a[n-i]
    return p,dp,ddp

def deflPoly(a,root): # Deflates a polynomial
    n = len(a)-1
    b = [(0.0 + 0.0j)]*n
    b[n-1] = a[n]

    for i in range(n-2,-1,-1):
        b[i] = a[i+1] + root*b[i+1]

    return b

def polyRoots(a,tol=1.0e-12):

    def laguerre(a,tol):
        x = random() # Starting value (random number)
        n = len(a) - 1

        for i in range(30):
            p,dp,ddp = eval_poly_derivatives(a,x)
            if abs(p) < tol: return x
            g = dp/p
            h = g*g - ddp/p
            f = cmath.sqrt((n - 1)*(n*h - g*g))
            if abs(g + f) > abs(g - f): dx = n/(g + f)
            else: dx = n/(g - f)
            x = x - dx
            if abs(dx) < tol: return x
        print('Too many iterations')

    n = len(a) - 1
    roots = np.zeros((n),dtype=complex)
    for i in range(n):
        x = laguerre(a,tol)
        if abs(x.imag) < tol: x = x.real
        roots[i] = x
        a = deflPoly(a,x)
    return roots