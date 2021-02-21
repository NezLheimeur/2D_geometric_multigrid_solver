#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 14:04:31 2021

@author: Nezar
"""

from math import pi, sin, cos, floor, sqrt
import pylab as plt
import numpy as np
import scipy as sp
import scipy.sparse as spa
import scipy.linalg as la
import copy
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import time


# Some paramters
_eps =1e-12
_maxiter=500

def _basic_check(A, b, x0):
    """ Common check for clarity """
    n, m = A.shape
    if(n != m):
        raise ValueError("Only square matrix allowed")
    if(b.size != n):
        raise ValueError("Bad rhs size")
    if (x0 is None):
        x0 = np.zeros(n)
    if(x0.size != n):
        raise ValueError("Bad initial value size")
    return x0


def laplace_2D(N, sigma, h):
    """ Construct the 2D laplace operator *
        input:  n: nb of inner grid points
                sigma:  weight of the none derived member of the equation
                h: size of the grid 
        output: A: 2D laplacian operator matrix
    """
    
    if(h<0): raise ValueError("h must be positive")
    
    n=int(sqrt(N))
    diagonals = []   
    diagonals = [-1*np.ones(n-1),(4+sigma*h*h)*np.ones(n),-1*np.ones(n-1)]
    B=spa.diags(diagonals, [-1,0,1]).toarray()
    I=(-1)*np.identity(n)
    
    A=spa.csr_matrix((N,N))
    for i in range(n):
        a, b, c = i*n, (i+1) * n, (i+2) * n
        A[a:b, a:b] = B
        if i < n-1: A[b:c, a:b]=A[a:b , b:c]=I
    A=1/h*h*A
    return A.todense()

def anisotropic_2D(N, epsilon, h):
    """ Construct the 2D anisotropic operator *
        input:  n: nb of inner grid points
                epsilon:  parameter fot the anisotropic equation on y dimension
                h: size of the grid 
        output: A: 2D anisotrpic operator matrix
    """
    
    if(h<0): raise ValueError("h must be positive")
    
    n=int(sqrt(N))
    diagonals = []   
    diagonals = [-1*epsilon*np.ones(n-1),2*(1+epsilon)*np.ones(n),-1*epsilon*np.ones(n-1)]
    B=spa.diags(diagonals, [-1,0,1]).toarray()
    I=-1*np.identity(n)
    
    A=spa.csr_matrix((N,N))
    for i in range(n):
        a, b, c = i*n, (i+1) * n, (i+2) * n
        A[a:b, a:b] = B
        if i < n-1: A[b:c, a:b]=A[a:b , b:c]=I
    A=1/h*h*A
    return A.todense()


def JOR(A, b, x0=None, omega=0.5, eps=_eps, maxiter=_maxiter):
    """
    Methode itérative stationnaire de sur-relaxation (Jacobi over relaxation)
    Convergence garantie si A est à diagonale dominante stricte
    A = D - E - F avec D diagonale, E (F) tri inf. (sup.) stricte
    Le préconditionneur est diagonal M = (1./omega) * D

    Output:
        - x is the solution at convergence or after maxiter iteration
        - residual_history is the norm of all residuals
    """
    x = _basic_check(A, b, x0)
    r = np.zeros(x.shape)
    residual_history = list()
    soluc_history = list()    
    iteration=0
    M=(1/omega)*(np.diag(A))
    r[:]=b- np.dot(A,x)
    norm_r0=la.norm(r)
    while(iteration<maxiter and la.norm(r)>eps*norm_r0):
        r[:]=b-np.dot(A,x)
        x+=r/M
        residual_history.append(la.norm(r))
        soluc_history.append(copy.copy(x))
        iteration+=1
        
    return x, residual_history, soluc_history


def SOR(A, b, x0=None, omega=1.5, eps=_eps, maxiter=_maxiter):
    """
    Methode itérative stationnaire de sur-relaxation successive
    (Successive Over Relaxation)

    A = D - E - F avec D diagonale, E (F) tri inf. (sup.) stricte
    Le préconditionneur est tri. inf. M = (1./omega) * D - E

    * Divergence garantie pour omega <= 0. ou omega >= 2.0
    * Convergence garantie si A est symétrique définie positive pour
    0 < omega  < 2.
    * Convergence garantie si A est à diagonale dominante stricte pour
    0 < omega  <= 1.

    Output:
        - x is the solution at convergence or after maxiter iteration
        - residual_history is the norm of all residuals

    """
    if (omega > 2.) or (omega < 0.):
        raise ArithmeticError("SOR will diverge")

    x = _basic_check(A, b, x0)
    r = np.zeros(x.shape)
    residual_history = list()
    soluc_history = list()    
    iteration=0
    M=(np.tril(A))
    M[np.diag_indices(A.shape[0])]/=omega
    r[:]=b- np.dot(A,x)
    norm_r0=la.norm(r)
    while(iteration<maxiter and la.norm(r)>eps*norm_r0):
        r[:]=b-np.dot(A,x)
        x+=la.solve_triangular(M,r,lower=True)
        residual_history.append(la.norm(r))
        soluc_history.append(copy.copy(x))
        iteration+=1
        
    return x, residual_history, soluc_history



def full_weighting_2D(N):
    """ 
    Classical full weightning injection,  
    The modification of coarse is done inplace
    
    input: N= nb of inner points in the square grid
    out
    """
    
    n=floor(sqrt(N))
    coarse_n = floor((n - 1) / 2)
    coarse_N = coarse_n * coarse_n
     
    
    R = spa.csr_matrix((coarse_N, N))
    k = 0
    for jy in range(1 , int(n), 2):
        for ix in range(1 , int(n), 2):
            fine_grid_k = (jy - 1) * n + ix
            
            R[k, fine_grid_k - n - 1] = 0.0625
            R[k, fine_grid_k - n   ]  = 0.125 
            R[k, fine_grid_k - n + 1] = 0.0625
            
            R[k, fine_grid_k - 1]     = 0.125
            R[k, fine_grid_k    ]     = 0.25
            R[k, fine_grid_k + 1]     = 0.125		

            R[k, fine_grid_k + n - 1] = 0.0625
            R[k, fine_grid_k + n    ] = 0.125
            R[k, fine_grid_k + n + 1] = 0.0625
            
            k = k + 1

    return R.todense()

def half_weighting_2D(N):
    """ 
    half weightning injection,  
    The modification of coarse is done inplace
    
    input: N= nb of inner points in the square grid
    out
    """
    
    n=floor(sqrt(N))
    coarse_n = floor((n - 1) / 2)
    coarse_N = coarse_n * coarse_n
     
    
    R = spa.csr_matrix((coarse_N, N))
    k = 0
    for jy in range(1 , int(n), 2):
        for ix in range(1 , int(n), 2):
            fine_grid_k = (jy - 1) * n + ix
            
            R[k, fine_grid_k - n - 1] = 0.0
            R[k, fine_grid_k - n   ]  = 0.125 
            R[k, fine_grid_k - n + 1] = 0.0625
            
            R[k, fine_grid_k - 1]     = 0.125
            R[k, fine_grid_k    ]     = 0.5
            R[k, fine_grid_k + 1]     = 0.125		

            R[k, fine_grid_k + n - 1] = 0.0
            R[k, fine_grid_k + n    ] = 0.125
            R[k, fine_grid_k + n + 1] = 0.0
            
            k = k + 1

    return R.todense()

def simple_injection_2D(N):
    """ 
    simple injection,  
    The modification of coarse is done inplace
    
    input: N= nb of inner points in the square grid
    out
    """
    
    n=floor(sqrt(N))
    coarse_n = floor((n - 1) / 2)
    coarse_N = coarse_n * coarse_n
     
    
    R = spa.csr_matrix((coarse_N, N))
    k = 0
    for jy in range(1 , int(n), 2):
        for ix in range(1 , int(n), 2):
            fine_grid_k = (jy - 1) * n + ix
            
            R[k, fine_grid_k - n - 1] = 0.0
            R[k, fine_grid_k - n   ]  = 0.0 
            R[k, fine_grid_k - n + 1] = 0.0
            
            R[k, fine_grid_k - 1]     = 0.0
            R[k, fine_grid_k    ]     = 1
            R[k, fine_grid_k + 1]     = 0.0		

            R[k, fine_grid_k + n - 1] = 0.0
            R[k, fine_grid_k + n    ] = 0.0
            R[k, fine_grid_k + n + 1] = 0.0
            
            k = k + 1

    return R.todense()


def interpolation_2D(N):
    """ 
    Classical linear interpolation full weighted (the modification of fine is done inplace)
    """
    return np.transpose(full_weighting_2D(N))*4


def plot(x, y, custom, label=""):
    """ 
    A custom plot function, usage: 
        f, ax = plot(x, y,'-x', label="u")
    """
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.plot(x, y, custom, label=label);     
    ax.set_xlabel(r"x")    
    ax.set_ylabel(r"u(x)")
    f.tight_layout()
    return f, ax

def plot_surface(x, y, z, color="blue", color_map=False, label=""):
    """ 
    A custom plot function, usage: 
        f, ax = plot(x, y,'-x', label="u")
    """
    f = plt.figure()
    ax = f.add_subplot(111, projection='3d')
    if(color_map==False):
        ax.plot_surface(x, y, z, color=color, label=label);   
    else:
        ax.plot_surface(x, y, z, color=color, cmap=cm.coolwarm, label=label);     
    ax.set_xlabel('X Axes')
    ax.set_ylabel('Y Axes')
    ax.set_zlabel('Z Axes')

    f.tight_layout()
    return f, ax


def tgcyc_2D(u_initial, nsegment=64, sigma_eps=0, b=np.zeros(64 -1), engine=SOR, restriction=simple_injection_2D, A_operator=laplace_2D, color_map=False, **kwargs):
    """ 
    Two grid cycle:
        - u_initial: initial guess
        -layer: nb of layers for the multigrid
        - nsegment: the number of segment per dimensions so that h = 1.0/nsegment
        - b: right hand side 
        - engine: the stationary iterative method used for smoothing 
        - sigma_eps: if laplacian problem: weight of the non differential member on the laplace equation/ if anisotropic problem is the epsilon, the weight of the second dimension
        - restriction: set up the kind of restriction you want full weightning, half weightning or simple injection
        - A_operator: set up the matrix operator for the problem (laplace_2D for a laplacian equation; anisotropic_2D for an anisotropic problem) the sigma_eps parameter varies according to the value choosen
        - plot_3D: false to not plot in 3D (by default), true to plot
        - color_map: True to plot the figure with a color map, False: to plot it in one color
    
    Warning: make the good distinction between the number of segments, the 
    number of nodes and the number of unknowns
    """
    if(nsegment%2): raise ValueError("nsegment must be even")
    
    # Beware that : nsegment
    # n = number of nodes per dimension 
    # n_inc = number of unknowns 
    n = nsegment + 1    
    h = 1.0 / nsegment
    H = 2. * h
    
    n_inc_h = (nsegment-1)*(nsegment-1)
    n_inc_H = ((nsegment/2)-1)*((nsegment/2)-1) 
    
    # Full points
    xh = np.linspace(0.,1., n) 
    xH = np.linspace(0.,1., n//2 + 1)    
    
   # Inner points
    xih = xh[1:-1]
    yih = xh[1:-1]
    xiH = xH[1:-1]
    yiH = xH[1:-1]
    
    Xh,Yh = np.meshgrid(xih,yih)
    
    #construct restriction and prolongation matrix
    I_hH=restriction(n_inc_h)   
    I_Hh=interpolation_2D(n_inc_h)
   
    # construction of Laplace operator 
    Ah = A_operator(n_inc_h, sigma_eps, h)

    if(restriction==full_weighting_2D):
        # construction of Laplace operator in coarse grid AH=IhH*Ah*IHh (Galerkin coarse grid possible because IhH=constant * transpose(IHh) condition is satisfied )
        AH = np.dot(I_hH,Ah)
        AH = np.dot(AH,I_Hh)
    else:
        AH = A_operator(n_inc_H, sigma_eps, H)
            
    # Initial guess    
    u0=u_initial     
    u0=u0.flatten()
    
    # set up right hand side
    b_h=b

    # Pre-smoothing Relaxation     
    u_smooth, residual_history, soluc_history = engine(Ah, b_h, x0=u0, omega=0.5, eps=_eps, maxiter=10)


    for xsol in soluc_history:
        plot_surface(Xh, Yh, np.reshape(u_smooth,(Xh.shape)), color_map=color_map)    
        
    #plt.show()
    
    # Compute the defect    
    d_h=copy.copy(b_h)
    d_h[:]-= np.ravel(np.dot(Ah,u_smooth))
    
    # Restriction with injection  
    d_H=np.ravel(np.dot(I_hH, d_h))
    
    # Solve on the coarse grid    
    e_H = la.solve(AH, d_H)
    
    # Prolongation  
    e_h=np.ravel(np.dot(I_Hh, e_H))
    
    # Update solution 
    u_smooth += e_h
    
    
    # Post-smoothing Relaxation 
    
    u_smooth_final, residual_history, soluc_history = engine(Ah, b_h, x0=u_smooth, omega=0.5, eps=_eps, maxiter=10)
    for xsol in soluc_history:
        plot_surface(Xh, Yh, np.reshape(u_smooth,(Xh.shape)), "red", color_map=color_map)    
                
    return u_smooth_final

def mgcyc_2D(u_initial, layer=5, nsegment=64, sigma_eps=0,  b=np.zeros(64 -1), engine=SOR, A_operator=laplace_2D, restriction=full_weighting_2D, plot_3D=False, color_map=False, **kwargs):
    """ 
    multigrid grid V-cycle:
        - u_initial: initial guess
        -layer: nb of layers for the multigrid
        - nsegment: the number of segment per dimensions so that h = 1.0/nsegment
        - b: right hand side 
        - engine: the stationary iterative method used for smoothing 
        - sigma_eps: if laplacian problem: weight of the non differential member on the laplace equation/ if anisotropic problem is the epsilon, the weight of the second dimension
        - restriction: set up the kind of restriction you want full weightning, half weightning or simple injection
        - A_operator: set up the matrix operator for the problem (laplace_2D for a laplacian equation; anisotropic_2D for an anisotropic problem) the sigma_eps parameter varies according to the value choosen
        - plot_3D: false to not plot in 3D (by default), true to plot
        - color_map: True to plot the figure with a color map, False: to plot it in one color
    
    
    Warning: make the good distinction between the number of segments, the 
    number of nodes and the number of unknowns
    """
    if(nsegment%2): raise ValueError("nsegment must be even")
    
    # Beware that : nsegment
    # n = number of nodes 
    # n_inc = number of unknowns
    
    print("layer: ", layer)
    
    injection_mode=restriction
    A_matrix=A_operator
    
    n_seg = nsegment    
    n = nsegment + 1    
    h = 1.0 / nsegment
    H = 2. * h
    
    n_inc_h = (nsegment-1)*(nsegment-1)
    n_inc_H = ((nsegment/2)-1)*((nsegment/2)-1) 
    
    # Full points
    xh = np.linspace(0.,1., n) 
    xH = np.linspace(0.,1., n//2 + 1)    
    
   # Inner points
    xih = xh[1:-1]
    yih = xh[1:-1]
    xiH = xH[1:-1]
    yiH = xH[1:-1]
    
    Xh,Yh = np.meshgrid(xih,yih)
    
    #construct restriction and prolongation matrix
    I_hH=injection_mode(n_inc_h) 
    I_Hh=interpolation_2D(n_inc_h)


    # construction of Laplace operator 
    Ah = A_matrix(n_inc_h, sigma_eps, h)

    if(injection_mode==full_weighting_2D):
        # construction of Laplace operator in coarse grid AH=IhH*Ah*IHh (Galerkin coarse grid possible because IhH=constant * transpose(IHh) condition is satisfied )       
        AH = np.dot(I_hH,Ah)
        AH = np.dot(AH,I_Hh)

    else:
        AH = A_matrix(n_inc_H, sigma_eps, H)
     
    # Initial guess    
    u0 = u_initial  
    u0=u0.flatten()
    
    #set up of the right hand side
    b_h =b

    # Pre-smoothing Relaxation     
    u_smooth, residual_history, soluc_history = engine(Ah, b_h, x0=u0, omega=0.5, eps=_eps, maxiter=10)
    if(plot_3D==True):    
        for xsol in soluc_history:
            plot_surface(Xh, Yh, np.reshape(u_smooth,(Xh.shape)), color_map=color_map)    
    
    # Compute the defect    
    d_h=copy.copy(b_h)    
    d_h[:]-= np.ravel(np.dot(Ah,u_smooth))
    
    # Restriction with injection  
    d_H=np.ravel(np.dot(I_hH, d_h))
       
    # Solve on the coarse grid  
    e_H = np.zeros(d_H.size)
            
    if(layer==1 or (n_inc_h-1)%2!=0 or (n_inc_h-1)<=7*7 ):
        e_H = la.solve(AH, d_H)
    else:
        e_H= mgcyc_2D(e_H, layer-1, n_seg/2, sigma_eps, d_H, restriction=injection_mode, A_operator=A_matrix)
        
    
    # Prolongation  
    e_h=np.ravel(np.dot(I_Hh, e_H))
    
    # Update solution 
    u_smooth += e_h
    
    
    # Post-smoothing Relaxation 
    u_smooth_final, residual_history, soluc_history = engine(Ah, b_h, x0=u_smooth, omega=0.5, eps=_eps, maxiter=10)
    
    if (plot_3D==True):
        for xsol in soluc_history:
            plot_surface(Xh, Yh, np.reshape(u_smooth,(Xh.shape)), "red", color_map=color_map)    
    
    print("e_h:", residual_history[-1])
    return u_smooth_final

def w_mgcyc_2D(u_initial, layer=5, nsegment=64, sigma_eps=0, b=np.zeros(64 -1), engine=SOR, restriction=full_weighting_2D, A_operator=laplace_2D, plot_3D=False, color_map=False, **kwargs):
    """ 
    multigrid-grid w-cycle:
        - u_initial: initial guess
        -layer: nb of layers for the multigrid
        - nsegment: the number of segment per dimensions so that h = 1.0/nsegment
        - b: right hand side 
        - engine: the stationary iterative method used for smoothing 
        - sigma_eps: if laplacian problem: weight of the non differential member on the laplace equation/ if anisotropic problem is the epsilon, the weight of the second dimension
        - restriction: set up the kind of restriction you want full weightning, half weightning or simple injection
        - A_operator: set up the matrix operator for the problem (laplace_2D for a laplacian equation; anisotropic_2D for an anisotropic problem) the sigma_eps parameter varies according to the value choosen
        - plot_3D: false to not plot in 3D (by default), true to plot
        - color_map: True to plot the figure with a color map, False: to plot it in one color
    
    
    Warning: make the good distinction between the number of segments, the 
    number of nodes and the number of unknowns
    """
    if(nsegment%2): raise ValueError("nsegment must be even")
    
    # Beware that : nsegment
    # n = number of nodes 
    # n_inc = number of unknowns
    
    print("layer: ", layer)

    injection_mode=restriction
    A_matrix=A_operator
            
    n_seg = nsegment    
    n = nsegment + 1    
    h = 1.0 / nsegment
    H = 2. * h
    
    n_inc_h = (nsegment-1)*(nsegment-1)
    n_inc_H = ((nsegment/2)-1)*((nsegment/2)-1) 
    
    # Full points
    xh = np.linspace(0.,1., n) 
    xH = np.linspace(0.,1., n//2 + 1)    
    
   # Inner points
    xih = xh[1:-1]
    yih = xh[1:-1]
    xiH = xH[1:-1]
    yiH = xH[1:-1]
    
    Xh,Yh = np.meshgrid(xih,yih)
    
    #construct restriction and prolongation matrix
    I_hH=injection_mode(n_inc_h)
    I_Hh=interpolation_2D(n_inc_h)

    # construction of Laplace operator 
    Ah = A_matrix(n_inc_h, sigma_eps, h)

    if(injection_mode==full_weighting_2D):
        # construction of Laplace operator in coarse grid AH=IhH*Ah*IHh (Galerkin coarse grid possible because IhH=constant * transpose(IHh) condition is satisfied )
        AH = np.dot(I_hH,Ah)
        AH = np.dot(AH,I_Hh)
    else:
        AH = A_matrix(n_inc_H, sigma_eps, H)
    
   
    # Initial guess    
    u0 = u_initial  
    u0=u0.flatten()
    
    #set up of the right hand side
    b_h =b

    # Pre-smoothing Relaxation     
    u_smooth, residual_history, soluc_history = engine(Ah, b_h, x0=u0, omega=0.5, eps=_eps, maxiter=10)
     
    if (plot_3D==True):
        for xsol in soluc_history:
            plot_surface(Xh, Yh, np.reshape(u_smooth,(Xh.shape)), color_map=color_map)    
    
    # Compute the defect    
    d_h=copy.copy(b_h)   
    d_h[:]-= np.ravel(np.dot(Ah,u_smooth))
    
    # Restriction with injection  
    d_H=np.ravel(np.dot(I_hH, d_h))
    
    
    # Solve on the coarse grid  
    e_H = np.zeros(d_H.size)
            
    if(layer==1 or (n_inc_h-1)%2!=0 or (n_inc_h-1)<=7*7 ):
        e_H = la.solve(AH, d_H)
    else:
        e_H = w_mgcyc_2D(e_H, layer-1, n_seg/2, sigma_eps, d_H, restriction=injection_mode, A_operator=A_matrix)
        
    
    # Prolongation  
    e_h=np.ravel(np.dot(I_Hh, e_H))
    
    # Update solution 
    u_smooth += e_h
    
    
    # Re-smoothing Relaxation  
    u_smooth_medium, residual_history, soluc_history = engine(Ah, b_h, x0=u_smooth, omega=0.5, eps=_eps, maxiter=10)
    
    #compute defect   
    d_h=copy.copy(b_h)    
    d_h[:]-= np.ravel(np.dot(Ah,u_smooth))
    
    # Restriction with injection   
    d_H=np.ravel(np.dot(I_hH, d_h))
   
    e_H = np.zeros(d_H.size)
    
    # Solve on the coarse grid   
    if(layer==1 or (n_inc_h-1)%2!=0 or (n_inc_h-1)<=7*7 ):
        e_H = la.solve(AH, d_H)
    else:
        e_H = w_mgcyc_2D(e_H, layer-1, n_seg/2, sigma_eps, d_H, restriction=injection_mode, A_operator=A_matrix)
    
    # Prolongation  
    e_h=np.ravel(np.dot(I_Hh, e_H))
    
    # Update solution 
    u_smooth += e_h
    
    # Post smoothing Relaxation 
    u_smooth_final, residual_history, soluc_history = engine(Ah, b_h, x0=u_smooth, omega=0.5, eps=_eps, maxiter=10)
    
    if (plot_3D==True):
        for xsol in soluc_history:
            plot_surface(Xh, Yh, np.reshape(u_smooth,(Xh.shape)),"red", color_map=color_map)    
    
    print("e_h:", residual_history[-1])    
    return u_smooth_final
 

# For debugging: 
#tgcyc_2D(nsegment=16)

nsegment=64
n=nsegment+1
n_inc=(nsegment-1)*(nsegment-1)
xh = np.linspace(0.,1., n)
xih = xh[1:-1]
yih = xh[1:-1]
Xh,Yh = np.meshgrid(xih,yih)
    
u0 = 0.5 * (np.sin(16. * Xh * pi) + np.sin(40. * Xh * pi))*(np.sin(16. * Yh * pi) + np.sin(40. * Yh * pi))    

u0 = 0 * (np.sin(16. * Xh * pi) + np.sin(40. * Xh * pi))*(np.sin(16. * Yh * pi) + np.sin(40. * Yh * pi))    

np.random.seed(10)
b = np.random.rand(n_inc)-0.5

sigma_values=[-100, -50, -20, -3, 0,1,3,5,10,20,50,100]
#injection_values=[simple_injection_2D]
#injection_values=[half_weighting_2D]
injection_values=[full_weighting_2D]

#you can try different type of injection weighting 
for injection in injection_values:
    print("injection type", injection)
    print("V-cycle")
    for sigma in sigma_values:
        print ("sigma = ",sigma)
        start_time = time.time()
        mgcyc_2D(u0, layer=15, nsegment=nsegment, b=b, sigma_eps=sigma, restriction=injection , A_operator=anisotropic_2D)
        end_time = time.time()
        print("execution time",end_time-start_time)

    print("W-cycle")
    for sigma in sigma_values:
        print ("sigma = ",sigma)
        start_time = time.time()
        w_mgcyc_2D(u0, layer=15, nsegment=nsegment, b=b, sigma_eps=sigma, restriction=injection, A_operator=anisotropic_2D)
        end_time = time.time()
        print("execution time",end_time-start_time)


 
# For real application 
#tgcyc_2D(u0, nsegment=64, A_operator=anisotropic_2D, sigma=0, b=b)