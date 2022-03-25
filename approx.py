from numbers import Integral
import numpy as np
import math
from .exceptions import RootError
from .anal import gaussPivot


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

def integrate(f, a, b, dx=0.01):

    x = np.arange(a, b, dx)
    integral = 0

    for i in x:
        integral += f(i + dx/2)*dx
    
    return(integral)

''' root = newtonRaphson(f,df,a,b,tol=1.0e-9).
Finds a root of f(x) = 0 by combining the Newton-Raphson
method with bisection. The root must be bracketed in (a,b).
Calls user-supplied functions f(x) and its derivative df(x).
'''
def newtonRaphson(f,df,a,b,tol=1.0e-9):

    fa = f(a)

    if fa == 0.0: return a

    fb = f(b)

    if fb == 0.0: return b

    if np.sign(fa) == np.sign(fb):
        RootError("Root not bracketed")

    x = 0.5*(a + b)
    
    for i in range(30):

        fx = f(x)

        if fx == 0.0: return x

        # Tighten the brackets on the root
        if np.sign(fa) != np.sign(fx): 
            b = x
        else: 
            a = x

        # Try a Newton-Raphson step
        dfx = df(x)

        # If division by zero, push x out of bounds
        with np.errstate(divide='ignore'):
            try: 
                dx = -fx/dfx
            except ZeroDivisionError: 
                dx = b - a

        x = x + dx

        # If the result is outside the brackets, use bisection
        if (b - x)*(x - a) < 0.0:
            dx = 0.5*(b - a)
            x = a + dx

        # Check for convergence
        if abs(dx) < tol*max(abs(b),1.0): return x

    print('Too many iterations in Newton-Raphson')


''' soln = newtonRaphson2(f,x,tol=1.0e-9).
Solves the simultaneous equations f(x) = 0 by
the Newton-Raphson method using {x} as the initial
guess. Note that {f} and {x} are vectors.
'''
def jacobian(f,x):
    h = 1.0e-4
    n = len(x)
    jac = np.zeros((n,n))
    f0 = f(x)
    for i in range(n):
        temp = x[i]
        x[i] = temp + h
        f1 = f(x)
        x[i] = temp
        jac[:,i] = (f1 - f0)/h
    return jac,f0

def newtonRaphson2(f,x,tol=1.0e-9):
    for i in range(30):

        jac,f0 = jacobian(f,x)

        if math.sqrt(np.dot(f0,f0)/len(x)) < tol: return x

        dx = gaussPivot(jac,-f0)
        x = x + dx

        if math.sqrt(np.dot(dx,dx)) < tol*max(max(abs(x)),1.0):
            return x

    print('Too many iterations in Newton Raphson 2')

''' Inew = trapezoid(f,a,b,Iold,k).
Recursive trapezoidal rule:
old = Integral of f(x) from x = a to b computed by
trapezoidal rule with 2ˆ(k-1) panels.
Inew = Same integral computed with 2ˆk panels.
'''
def trapezoid(f,a,b,Iold,k):
    if k == 1:Inew = (f(a) + f(b))*(b - a)/2.0
    else:
        n = 2**(k -2 ) # Number of new points
        h = (b - a)/n # Spacing of new points
        x = a + h/2.0
        sum = 0.0
        for i in range(n):
            sum = sum + f(x)
            x=x+h
        Inew = (Iold + h*sum)/2.0
    return Inew

''' x,A = gaussNodes(m,tol=10e-9)
Returns nodal abscissas {x} and weights {A} of
Gauss-Legendre m-point quadrature.
'''
def gaussNodes(m,tol=10e-9):

    def legendre(t,m):
        p0 = 1.0; p1 = t
        for k in range(1,m):
            p = ((2.0*k + 1.0)*t*p1 - k*p0)/(1.0 + k )
            p0 = p1; p1 = p
        dp = m*(p0 - t*p1)/(1.0 - t**2)
        return p,dp

    A = np.zeros(m)
    x = np.zeros(m)
    nRoots = int((m + 1)/2) # Number of non-neg. roots

    for i in range(nRoots):
        t = math.cos(math.pi*(i + 0.75)/(m + 0.5))# Approx. root

        for j in range(30):
            p,dp = legendre(t,m) # Newton-Raphson
            dt = -p/dp; t = t + dt # method

            if abs(dt) < tol:
                x[i] = t; x[m-i-1] = -t
                A[i] = 2.0/(1.0 - t**2)/(dp**2) # Eq.(6.25)
                A[m-i-1] = A[i]
                break
    return x,A

''' I = gaussQuad(f,a,b,m).
    Computes the integral of f(x) from x = a to b
    with Gauss-Legendre quadrature using m nodes.
'''
def gaussQuad(f,a,b,m):
    c1 = (b + a)/2.0
    c2 = (b - a)/2.0
    x,A = gaussNodes(m)
    sum = 0.0
    for i in range(len(x)):
        sum = sum + A[i]*f(c1 + c2*x[i])
    return c2*sum

''' I = gaussQuad2(f,xc,yc,m).
Gauss-Legendre integration of f(x,y) over a
quadrilateral using integration order m.
{xc},{yc} are the corner coordinates of the quadrilateral.
'''
def gaussQuad2(f,x,y,m):
    def jac(x,y,s,t):
        J = np.zeros((2,2))
        J[0,0] = -(1.0 - t)*x[0] + (1.0 - t)*x[1] \
                    + (1.0 + t)*x[2] - (1.0 + t)*x[3]
        J[0,1] = -(1.0 - t)*y[0] + (1.0 - t)*y[1] \
                    + (1.0 + t)*y[2] - (1.0 + t)*y[3]
        J[1,0] = -(1.0 - s)*x[0] - (1.0 + s)*x[1] \
                    + (1.0 + s)*x[2] + (1.0 - s)*x[3]
        J[1,1] = -(1.0 - s)*y[0] - (1.0 + s)*y[1] \
                    + (1.0 + s)*y[2] + (1.0 - s)*y[3]
        return (J[0,0]*J[1,1] - J[0,1]*J[1,0])/16.0
    
    def map(x,y,s,t):
        N = np.zeros(4)
        N[0] = (1.0 - s)*(1.0 - t)/4.0
        N[1] = (1.0 + s)*(1.0 - t)/4.0
        N[2] = (1.0 + s)*(1.0 + t)/4.0
        N[3] = (1.0 - s)*(1.0 + t)/4.0
        xCoord = np.dot(N,x)
        yCoord = np.dot(N,y)
        return xCoord,yCoord
    
    s,A = gaussNodes(m)
    sum = 0.0
    for i in range(m):
        for j in range(m):
            xCoord,yCoord = map(x,y,s[i],s[j])
            sum = sum + A[i]*A[j]*jac(x,y,s[i],s[j]) \
                        *f(xCoord,yCoord)
    return sum

''' integral = triangleQuad(f,xc,yc).
Integration of f(x,y) over a triangle using
the cubic formula.
{xc},{yc} are the corner coordinates of the triangle.
'''
def triangleQuad(f,xc,yc):
    alpha = np.array([[1/3, 1/3.0, 1/3], \
                        [0.2, 0.2, 0.6], \
                        [0.6, 0.2, 0.2], \
                        [0.2, 0.6, 0.2]])
    W = np.array([-27/48,25/48,25/48,25/48])
    x = np.dot(alpha,xc)
    y = np.dot(alpha,yc)
    A = (xc[1]*yc[2] - xc[2]*yc[1] - xc[0]*yc[2] + xc[2]*yc[0] + xc[0]*yc[1] - xc[1]*yc[0])/2.0
    sum = 0.0

    for i in range(4):
        sum = sum + W[i] * f(x[i],y[i])

    return A*sum

''' X,Y = kut_int(F,x,y,xStop,h).
4th-order Runge-Kutta method for solving the
initial value problem {y}' = {F(x,{y})}, where
{y} = {y[0],y[1],...y[n-1]}.
x,y = initial conditions
xStop = terminal value of x
h = increment of x used in integration
F = user-supplied function that returns the
array F(x,y) = {y'[0],y'[1],...,y'[n-1]}.
'''
def kutint(F,x,y,xStop,h):

    def run_kut4(F,x,y,h):
        K0 = h*F(x,y)
        K1 = h*F(x + h/2.0, y + K0/2.0)
        K2 = h*F(x + h/2.0, y + K1/2.0)
        K3 = h*F(x + h, y + K2)
        return (K0 + 2.0*K1 + 2.0*K2 + K3)/6.0

    X = []
    Y = []
    X.append(x)
    Y.append(y)

    while x < xStop:
        h = min(h,xStop - x)
        y = y + run_kut4(F,x,y,h)
        x=x+h
        X.append(x)
        Y.append(y)

    return np.array(X),np.array(Y)

''' yStop = integrate (F,x,y,xStop,tol=1.0e-6)
Modified midpoint method for solving the
initial value problem y' = F(x,y}.
x,y = initial conditions
xStop = terminal value of x
yStop = y(xStop)
F = user-supplied function that returns the
array F(x,y) = {y'[0],y'[1],...,y'[n-1]}.
'''
def midint(F,x,y,xStop,tol):
    
    def midpoint(F,x,y,xStop,nSteps):
        # Midpoint formulas
        h = (xStop - x)/nSteps
        y0 = y
        y1 = y0 + h*F(x,y0)
        for i in range(nSteps-1):
            x=x+h
            y2 = y0 + 2.0*h*F(x,y1)
            y0 = y1
            y1 = y2
        return 0.5*(y1 + y0 + h*F(x,y2))

    def richardson(r,k):
        # Richardson’s extrapolation
        for j in range(k-1,0,-1):
            const = (k/(k - 1.0))**(2.0*(k-j))
            r[j] = (const*r[j+1] - r[j])/(const - 1.0)
        return

    kMax = 51
    n = len(y)
    r = np.zeros((kMax,n))

    # Start with two integration steps
    nSteps = 2
    r[1] = midpoint(F,x,y,xStop,nSteps)
    r_old = r[1].copy()

    # Increase the number of integration points by 2
    # and refine result by Richardson extrapolation
    for k in range(2,kMax):
        nSteps = 2*k
        r[k] = midpoint(F,x,y,xStop,nSteps)
        richardson(r,k)

        # Compute RMS change in solution
        e = math.sqrt(np.sum((r[1] - r_old)**2)/n)

        # Check for convergence
        if e < tol: return r[1]
        r_old = r[1].copy()

    print("Midpoint method did not converge")

''' X,Y = bulStoer(F,x,y,xStop,H,tol=1.0e-6).
Simplified Bulirsch-Stoer method for solving the
initial value problem {y}' = {F(x,{y})}, where
{y} = {y[0],y[1],...y[n-1]}.
x,y = initial conditions
xStop = terminal value of x
H = increment of x at which results are stored
F = user-supplied function that returns the
array F(x,y) = {y'[0],y'[1],...,y'[n-1]}.
'''
def bulStoer(F,x,y,xStop,H,tol=1.0e-6):
    X = []
    Y = []
    X.append(x)
    Y.append(y)
    while x < xStop:
        H = min(H,xStop - x)
        y = midint(F,x,y,x + H,tol) # Midpoint method
        x=x+H
        X.append(x)
        Y.append(y)

    return(np.array(X), np.array(Y))
