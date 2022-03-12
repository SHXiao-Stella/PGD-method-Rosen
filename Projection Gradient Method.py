# Rosen Gradient Projection Method
# Ref:《最优化理论与算法》陈宝林，Chapter 12
# Author: Stella

import numpy as np
from scipy.optimize import minimize_scalar

#minmize problem
#min 2*x[0]**2 + 2*x[1]**2 -2*x[0]*x[1] - 4*x[0] - 6*x[1]
#s.t.
# -x[0]-x[1]>=-2,
# -x[0]-5*x[1]>=-5,
# x[0]>=0, x[1]>=0

n=2
#Coefficient matrix
A = np.array([[-1, -1], [-1, -5], [1, 0], [0, 1]])
b = np.array([-2, -5, 0, 0]).reshape((1, len(A)))

#Objective function
def fun(x):
    ff = 2*x[0]**2 + 2*x[1]**2 -2*x[0]*x[1] - 4*x[0] - 6*x[1]
    return ff

#left hand side
def func_eval(x, A, b):
    con_left = A @ x - b
    return con_left

def gradient(x):
    g_1 = 4*x[0]-2*x[1]-4
    g_2 = 4*x[1]-2*x[0]-6
    gg  = np.array([g_1,g_2])
    return gg

def func(lambda_o):
    find_min_f = fun(x + lambda_o * d)
    find_min_f = np.int(find_min_f)
    return find_min_f

x=np.array([0,0])

number = 1
while number < 10000:
    con_left = func_eval(x,A,b)
    suoyin_1 = np.where(con_left == 0)[1]
    suoyin_2 = np.where(con_left >0)[1]
    A1 = A[suoyin_1]
    A2 = A[suoyin_2]
    b1 = np.transpose(np.transpose(b)[suoyin_1])
    b2 = np.transpose(np.transpose(b)[suoyin_2])

    N = A1.T @ (np.linalg.inv(A1 @ A1.T)) @ A1
    I = np.eye(len(N))
    P = I - N
    G = gradient(x)
    d = -1 * P @ G

    count = 1
    while count < 100:
        if d.any() == 0 or all([v < 1e-5 for v in d]):
            W = (np.linalg.inv(A1 @ A1.T)) @ A1 @ G
            if (W>=0).all():
                print('The K-T point is', x)
                judge = 1
                break
            else:
                print('Continue searching')
                judge = 0
                suoyin_3 = np.argmin(W)
                A1 = np.delete(A1, suoyin_3, axis=0)
                N = A1.T @ (np.linalg.inv(A1 @ A1.T)) @ A1
                I = np.eye(len(N))
                P = I - N
                G = gradient(x)
                d = -1 * P @ G
                count = count + 1
        else:
            break
        d = d

    b_hat = b2 - A2 @ x
    d_hat = A2 @ d
    d_hat_u = d_hat.reshape((1, len(d_hat)))
    d_hat_0 = np.array(np.where(d_hat < 0))
    lieshu = (d_hat_0.shape)[1]
    lamb = []
    for i in range(lieshu):
        k = d_hat_0[0, i]
        lam = b_hat[0, k] / d_hat_u[0, k]
        lamb.append(lam)
    lamb = np.array(lamb, dtype=float)
    lambda_max = np.min(lamb)
    a_bound = 0

    ## if the objective function is not concave,convex or expressed:
    # search = 50000
    # fi = []
    # for i in range(search):
    #     ki = i * 1/search
    #     fii = fun(x+ki*lambda_max*d)
    #     fi.append(fii)
    #     i = i+1
    # fi = np.array(fi, dtype = float)
    # suoyin_4 = np.argmin(fi)
    # lambda_op = (1+suoyin_4) * (1/search) *lambda_max

    #if the objective function is normal
    res = minimize_scalar(fun, bracket=(a_bound, lambda_max), bounds=(a_bound, lambda_max), args=(),
                                method='bounded', tol=None, options=None)
    lambda_op = res.x

    if judge ==1:
        break
        print('The K-T point is', x)
    else:
        number = number+1
        x = x + lambda_op * d

print('The optimal solution is',x)