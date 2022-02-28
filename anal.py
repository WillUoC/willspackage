## Wills Numerical Analysis Package

import numpy as np
import math
import matplotlib.pyplot as plt
from .linalg import gaussPivot

''' c,d,e = LUdecomp3(c,d,e).
LU decomposition of tridiagonal matrix [c\d\e]. On output
{c},{d} and {e} are the diagonals of the decomposed matrix.
62 Systems of Linear Algebraic Equations
x = LUsolve(c,d,e,b).
Solves [c\d\e]{x} = {b}, where {c}, {d} and {e} are the
vectors returned from LUdecomp3.
'''
def __LUdecomp3(c,d,e):
    n = len(d)

    for k in range(1,n):
        lam = c[k-1]/d[k-1]
        d[k] = d[k] - lam*e[k-1]
        c[k-1] = lam

    return c,d,e

def __LUsolve3(c,d,e,b):
    n = len(d)

    for k in range(1,n):
        b[k] = b[k] - c[k-1]*b[k-1]

    b[n-1] = b[n-1]/d[n-1]
    for k in range(n-2,-1,-1):
        b[k] = (b[k] - e[k]*b[k+1])/d[k]

    return b

''' p = evalPoly(a,xData,x).
Evaluates Newton's polynomial p at x. The coefficient
vector {a} can be computed by the function 'coeffts'.
a = coeffts(xData,yData).
Computes the coefficients of Newton's polynomial.
'''
def poly_fit(a,xData,x):
    n = len(xData) - 1 # Degree of polynomial
    p = a[n]

    for k in range(1,n+1):
        p = a[n-k] + (x -xData[n-k])*p

    return p

''' a = __get_coefficients(xData, yData)
Internal Function; Computes a coefficient vector {a} given the plot data
'''
def get_coefficients(xData, yData):
    m = len(xData) # Number of data points
    a = yData.copy()

    for k in range(1,m):
        a[k:m] = (a[k:m] - a[k-1])/(xData[k:m] - xData[k-1])

    return a

''' p = rational(xData,yData,x)
Evaluates the diagonal rational function interpolant p(x)
that passes through the data points
'''
def evaluate_rational(xData,yData,x):
    m = len(xData)
    r = yData.copy()
    rOld = np.zeros(m)

    for k in range(m-1):
        for i in range(m-k-1):

            if abs(x - xData[i+k+1]) < 1.0e-9:
                return yData[i+k+1]

            else:
                c1 = r[i+1] - r[i]
                c2 = r[i+1] - rOld[i+1]
                c3 = (x - xData[i])/(x - xData[i+k+1])
                r[i] = r[i+1] + c1/(c3*(1.0 - c1/c2) - 1.0)
                rOld[i+1] = r[i+1]

    return r[0]

''' k = curvatures(xData,yData).
Returns the curvatures of cubic spline at its knots.
y = eval_spline(xData,yData,k,x).
Evaluates cubic spline at x. The curvatures k can be
computed with the function 'curvatures'.
'''
def curvatures(xData,yData):
    n = len(xData) - 1
    c = np.zeros(n)
    d = np.ones(n+1)
    e = np.zeros(n)
    k = np.zeros(n+1)

    c[0:n-1] = xData[0:n-1] - xData[1:n]
    d[1:n] = 2.0*(xData[0:n-1] - xData[2:n+1])
    e[1:n] = xData[1:n] - xData[2:n+1]
    k[1:n] =6.0*(yData[0:n-1] - yData[1:n]) / (xData[0:n-1] - xData[1:n]) - 6.0*(yData[1:n] - yData[2:n+1]) / (xData[1:n] - xData[2:n+1])
    
    __LUdecomp3(c,d,e)
    __LUsolve3(c,d,e,k)

    return k

''' iLeft = __find_segment(xData, x)
Evaluates a polynomial and finds the left edge of the segment
'''
def __find_segment(xData,x):
    iLeft = 0
    iRight = len(xData)- 1

    while 1:
        if (iRight-iLeft) <= 1: return iLeft

        i = (iLeft + iRight)//2

        if x < xData[i]: iRight = i
        else: iLeft = i

'''y = eval_spline(xData, yData, k, x)
Returns the y value associated with the given x, for a polynomial fit on the data given
'''
def eval_spline(xData,yData,k,x):
    i = __find_segment(xData,x)
    h = xData[i] - xData[i+1]

    y = ((x - xData[i+1])**3/h - (x - xData[i+1])*h)*k[i]/6.0 - ((x - xData[i])**3/h - (x - xData[i])*h)*k[i+1]/6.0 + (yData[i]*(x - xData[i+1]) - yData[i+1]*(x - xData[i]))/h
    
    return y

''' c = fit_poly(xData,yData,m).
Returns coefficients of the polynomial
p(x) = c[0] + c[1]x + c[2]x^2 +...+ c[m]x^m
that fits the specified data in the least
squares sense.
sigma = stdDev(c,xData,yData).
Computes the std. deviation between p(x)
and the data.
'''
def fit_reg_poly(xData,yData,m):
    a = np.zeros((m+1,m+1))
    b = np.zeros(m+1)
    s = np.zeros(2*m+1)

    for i in range(len(xData)):
        temp = yData[i]

        for j in range(m+1):
            b[j] = b[j] + temp
            temp = temp*xData[i]

        temp = 1.0

        for j in range(2*m+1):
            s[j] = s[j] + temp
            temp = temp*xData[i]

    for i in range(m+1):
        for j in range(m+1):
            a[i,j] = s[i+j]
    
    return gaussPivot(a,b)

''' p = __evaluate_polynomial(c, x)
Internal function; Evaluates polynomial p at x
'''
def eval_reg_polynomial(c,x):
    m = len(c) - 1
    p = c[m]

    for j in range(m):
        p = p*x + c[m-j-1]

    return p

'''sigma = stdDev(c, xData, yData)
Return the standard deviations given coefficients and plot data
'''
def stdDev(c,xData,yData):

    n = len(xData) - 1
    m = len(c) - 1
    sigma = 0.0

    for i in range(n+1):
        p = eval_reg_polynomial(c,xData[i])
        sigma = sigma + (yData[i] - p)**2

    sigma = math.sqrt(sigma/(n - m))
    return sigma

''' plot_polynomial(xData,yData,coeff,xlab='x',ylab='y')
Plots data points and the fitting
polynomial defined by its coefficient
array coeff = [a0, a1. ...]
xlab and ylab are optional axis labels
'''
def plot_polynomial(xData,yData,coeff,xlab='x',ylab='y'):
    m = len(coeff)
    x1 = min(xData)
    x2 = max(xData)
    dx = (x2 - x1)/20.0
    x = np.arange(x1,x2 + dx/10.0,dx)
    y = np.zeros((len(x)))*1.0

    for i in range(m):
        y = y + coeff[i]*x**i
    
    plt.plot(xData,yData,'o',x,y,'-')
    plt.xlabel(xlab); plt.ylabel(ylab)
    plt.grid (True)
    plt.show()