import numpy as np
from numpy.linalg import det
import math
import sys

from .funcs import err

""" swapRows(v,i,j).
    Swaps rows i and j of a vector or matrix [v].
    swapCols(v,i,j).
    Swaps columns of matrix [v].
"""
def swapRows(v,i,j):
    if len(v.shape) == 1:
        v[i],v[j] = v[j],v[i]
    else:
        v[[i,j],:] = v[[j,i],:]

def swapCols(v,i,j):
    v[:,[i,j]] = v[:,[j,i]]

""" x = gaussPivot(a,b,tol=1.0e-12).
Solves [a]{x} = {b} by Gauss elimination with
scaled row pivoting
"""
def gaussPivot(a,b,tol=1.0e-12):
    n = len(b)

    # Set up scale factors
    s = np.zeros(n)
    for i in range(n):
        s[i] = max(np.abs(a[i,:]))

    for k in range(0,n-1):

        # Row interchange, if needed
        p = np.argmax(np.abs(a[k:n,k])/s[k:n]) + k
        if abs(a[p,k]) < tol: err('Matrix is singular')
        if p != k:
            swapRows(b,k,p)
            swapRows(s,k,p)
            swapRows(a,k,p)


        # Elimination
        for i in range(k+1,n):
            if a[i,k] != 0.0:
                lam = a[i,k]/a[k,k]
                a[i,k+1:n] = a[i,k+1:n] - lam*a[k,k+1:n]
                b[i] = b[i] - lam*b[k]

    if abs(a[n-1,n-1]) < tol: err('Matrix is singular')
    
    # Back substitution
    b[n-1] = b[n-1]/a[n-1,n-1]
    for k in range(n-2,-1,-1):
        b[k] = (b[k] - np.dot(a[k,k+1:n],b[k+1:n]))/a[k,k]
    return b

def cramers_rule(a, b):
    """ x = cramers_rule(a, b)
        performs cramers rule to solve a non-singular set of liear equations.
    """
    x = np.zeros((a.shape[1], 1))

    for i in range(0, a.shape[1]):
        m = a.copy()
        for k in range(0, a.shape[0]):
            m[k, i] = b[k]

        x[i] = (det(m)/det(a))

    return(x)

""" a,seq = LUdecomp(a,tol=1.0e-9).
LU decomposition of matrix [a] using scaled row pivoting.
The returned matrix [a] = contains [U] in the upper
triangle and the nondiagonal terms of [L] in the lower triangle.
Note that [L][U] is a row-wise permutation of the original [a];
the permutations are recorded in the vector {seq}.

x = LUsolve(a,b,seq).
Solves [L][U]{x} = {b}, where the matrix [a] = and the
permutation vector {seq} are returned from LUdecomp.
"""
def LUdecomp(a,tol=1.0e-9):
    n = len(a)
    seq = np.array(range(n))

    # Set up scale factors
    s = np.zeros((n))
    for i in range(n):
        s[i] = max(abs(a[i,:]))


    for k in range(0,n-1):
        # Row interchange, if needed
        p = np.argmax(np.abs(a[k:n,k])/s[k:n]) + k
        if abs(a[p,k]) < tol: err('Matrix is singular')
        if p != k:
            swapRows(s,k,p)
            swapRows(a,k,p)
            swapRows(seq,k,p)
        # Elimination
        for i in range(k+1,n):
            if a[i,k] != 0.0:
                lam = a[i,k]/a[k,k]
                a[i,k+1:n] = a[i,k+1:n] - lam*a[k,k+1:n]
                a[i,k] = lam
    return a,seq

def LUfullsolve(a,b):
    """x = LUfullsolve(a, b)
        Performs the actions of LUdecomp and LUsolve
    """
    a, seq = LUdecomp(a.copy())

    n = len(a)

    # Rearrange constant vector; store it in [x]
    x = b.copy()
    for i in range(n):
        x[i] = b[seq[i]]

    # Solution
    for k in range(1,n):
        x[k] = x[k] - np.dot(a[k,0:k],x[0:k])
    x[n-1] = x[n-1]/a[n-1,n-1]

    for k in range(n-2,-1,-1):
        x[k] = (x[k] - np.dot(a[k,k+1:n],x[k+1:n]))/a[k,k]
    return x

def LUsolve(a,b, seq):
    """x = LUsolve(a, b, seq)
        solves a linear set of equations
    """
    n = len(a)

    # Rearrange constant vector; store it in [x]
    x = b.copy()
    for i in range(n):
        x[i] = b[seq[i]]

    # Solution
    for k in range(1,n):
        x[k] = x[k] - np.dot(a[k,0:k],x[0:k])
    x[n-1] = x[n-1]/a[n-1,n-1]

    for k in range(n-2,-1,-1):
        x[k] = (x[k] - np.dot(a[k,k+1:n],x[k+1:n]))/a[k,k]
    return x

def check_accuracy(a, b, x):
    """mean_accuracy = check_accuracy(a, b, x)
        checks the mean accuracy when given a linear set of equations and their answers.
    """
#    acc = np.zeros(a.shape[0])
#    for i in range(0, a.shape[0]):
#        acc[i] = np.dot(a[i], x) - b[i]
#    return(np.mean(acc))
    return(np.mean(np.dot(a,x)-b))
'''a, b = random_eqns(n)
    int n -- number of equations

    a -> nxn matrix (LHS)
    b -> 1xn matrix (RHS)
'''
def random_eqns(n):
    return(np.random.randint(-100, 100, (n, n)), np.random.randint(-100, 100, (n, 1)))

def matInv(a):
    """a_inverse = matInv(a)
        Returns the inverse matrix given a matrix
    """
    n = len(a[0])
    aInv = np.identity(n)
    a,seq = LUdecomp(a)
    for i in range(n):
        aInv[:,i] = LUsolve(a,aInv[:,i],seq)
    return aInv

''' x,numIter,omega = gaussSeidel(iterEqs,x,tol = 1.0e-9)
Gauss-Seidel method for solving [A]{x} = {b}.
The matrix [A] should be sparse. User must supply the
function iterEqs(x,omega) that returns the improved {x},
given the current {x} ('omega' is the relaxation factor).
'''
def gaussSeidel(iterEqs,x,tol = 1.0e-9):

    omega = 1.0
    k = 10
    p=1

    for i in range(1,501):
        xOld = x.copy()
        x = iterEqs(x,omega)
        dx = math.sqrt(np.dot(x-xOld,x-xOld))
        if dx < tol: return x,i,omega

        # Compute relaxation factor after k+p iterations
        if i == k: dx1 = dx

        if i == k + p:
            dx2 = dx
            omega = 2.0/(1.0 + math.sqrt(1.0 \
                    - (dx2/dx1)**(1.0/p)))

    print('Gauss-Seidel failed to converge')

''' x, numIter = conjGrad(Av,x,b,tol=1.0e-9)
Conjugate gradient method for solving [A]{x} = {b}.
The matrix [A] should be sparse. User must supply
the function Av(v) that returns the vector [A]{v}.
'''
def conjGrad(Av,x,b,tol=1.0e-9):

    n = len(b)
    r = np.subtract(b, Av(x))
    s = r.copy()

    for i in range(n*10):
        u = Av(s)
        alpha = np.dot(s,r)/np.dot(s,u)
        x = x + alpha*s
        
        r = b - Av(x)

        if(abs(np.mean(r))) < tol:
            break
        else:
            beta = -np.dot(r,u)/np.dot(s,u)
            s = r + beta*s
    
    return x,i