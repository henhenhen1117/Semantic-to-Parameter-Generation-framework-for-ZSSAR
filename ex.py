# import numpy as np
# import time
#
# # Rosenbrock function
# def f(x):
#     return sum((1-x[i])**2 + 100*(x[i+1]-x[i]**2)**2 for i in range(9))
#
# def grad(x):
#     g = np.zeros(10)
#     g[0] = -2*(1-x[0]) - 400*x[0]*(x[1]-x[0]**2)
#     for i in range(1,9):
#         g[i] = (-2*(1-x[i])
#                 - 400*x[i]*(x[i+1]-x[i]**2)
#                 + 200*(x[i]-x[i-1]**2))
#     g[9] = 200*(x[9]-x[8]**2)
#     return g
#
# def hess(x):
#     H = np.zeros((10,10))
#     for i in range(9):
#         H[i,i] += 202 - 400*(x[i+1]-3*x[i]**2)
#         H[i,i+1] = H[i+1,i] = -400*x[i]
#     H[9,9] = 200
#     return H
#
# # Steepest descent
# x = np.ones(10)*2
# start = time.time()
# for k in range(10):
#     g = grad(x)
#     alpha = 1e-3
#     x = x - alpha*g
# print("SD f =", f(x), "CPU =", time.time()-start)
#
# # Damped Newton
# x = np.ones(10)*2
# start = time.time()
# for k in range(10):
#     g = grad(x)
#     H = hess(x)
#     d = -np.linalg.solve(H, g)
#     lam = 1.0
#     c = 1e-4
#     while f(x+lam*d) > f(x) + c*lam*np.dot(g,d):
#         lam *= 0.5
#     x = x + lam*d
# print("Newton f =", f(x), "CPU =", time.time()-start)
# import numpy as np
# import time
#
# # ======================================================
# # Rosenbrock function (10D)
# # f(x) = sum_{i=1}^9 [(1-x_i)^2 + 100(x_{i+1}-x_i^2)^2]
# # ======================================================
#
# def f(x):
#     return sum((1 - x[i])**2 + 100*(x[i+1] - x[i]**2)**2 for i in range(9))
#
# def grad(x):
#     g = np.zeros(10)
#     g[0] = -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2)
#     for i in range(1, 9):
#         g[i] = (-2*(1 - x[i])
#                 - 400*x[i]*(x[i+1] - x[i]**2)
#                 + 200*(x[i] - x[i-1]**2))
#     g[9] = 200*(x[9] - x[8]**2)
#     return g
#
# def hess(x):
#     H = np.zeros((10, 10))
#     for i in range(9):
#         H[i, i] += 202 - 400*(x[i+1] - 3*x[i]**2)
#         H[i, i+1] = -400*x[i]
#         H[i+1, i] = -400*x[i]
#     H[9, 9] = 200
#     return H
#
# # ======================================================
# # Steepest Descent Method
# # ======================================================
#
# print("========== Steepest Descent Method ==========")
#
# x = np.ones(10) * 2
# alpha = 1e-3
# start = time.time()
#
# print("k    f(x_k)              CPU(s)")
# for k in range(10):
#     g = grad(x)
#     x = x - alpha * g
#     cpu = time.time() - start
#     print(f"{k+1:<2}   {f(x):<18.10f}   {cpu:.6f}")
#
# print("\nFinal f(x) =", f(x))
# print("Total CPU time =", time.time() - start)
#
# # ======================================================
# # Damped Newton Method (Armijo line search)
# # ======================================================
#
# print("\n========== Damped Newton Method ==========")
#
# x = np.ones(10) * 2
# start = time.time()
#
# beta = 0.5
# c = 1e-4
#
# print("k    f(x_k)              lambda_k    CPU(s)")
# for k in range(10):
#     g = grad(x)
#     H = hess(x)
#     d = -np.linalg.solve(H, g)
#
#     lam = 1.0
#     while f(x + lam*d) > f(x) + c * lam * np.dot(g, d):
#         lam *= beta
#
#     x = x + lam * d
#     cpu = time.time() - start
#     print(f"{k+1:<2}   {f(x):<18.10f}   {lam:<10.4f}   {cpu:.6f}")
#
# print("\nFinal f(x) =", f(x))
# print("Total CPU time =", time.time() - start)
# import numpy as np
# import time
#
# # ======================================================
# # Rosenbrock function (10D)
# # f(x) = sum_{i=1}^9 [(1-x_i)^2 + 100(x_{i+1}-x_i^2)^2]
# # ======================================================
#
# def f(x):
#     return sum((1 - x[i])**2 + 100*(x[i+1] - x[i]**2)**2 for i in range(9))
#
# def grad(x):
#     g = np.zeros(10)
#     g[0] = -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2)
#     for i in range(1, 9):
#         g[i] = (-2*(1 - x[i])
#                 - 400*x[i]*(x[i+1] - x[i]**2)
#                 + 200*(x[i] - x[i-1]**2))
#     g[9] = 200*(x[9] - x[8]**2)
#     return g
#
# def hess(x):
#     H = np.zeros((10, 10))
#     for i in range(9):
#         H[i, i] += 202 - 400*(x[i+1] - 3*x[i]**2)
#         H[i, i+1] = -400*x[i]
#         H[i+1, i] = -400*x[i]
#     H[9, 9] = 200
#     return H
#
# # ======================================================
# # Exact Line Search (Golden Section Method)
# # ======================================================
#
# def exact_line_search(x, d, tol=1e-5):
#     """
#     Solve: min_{lambda > 0} f(x + lambda * d)
#     """
#     a, b = 0.0, 1.0
#     phi = lambda lam: f(x + lam * d)
#
#     # Expand interval to ensure minimum is inside [a, b]
#     while phi(b) < phi(a):
#         b *= 2
#
#     rho = 0.618
#     while b - a > tol:
#         l = b - rho * (b - a)
#         r = a + rho * (b - a)
#         if phi(l) < phi(r):
#             b = r
#         else:
#             a = l
#
#     return 0.5 * (a + b)
#
# # ======================================================
# # Steepest Descent Method (Exact Line Search)
# # ======================================================
#
# print("========== Steepest Descent Method (Exact Line Search) ==========")
#
# x = np.ones(10) * 2
# eps = 1e-6
# max_iter = 10
#
# start = time.time()
# print("k    f(x_k)              CPU(s)")
#
# for k in range(max_iter):
#     g = grad(x)
#     if np.linalg.norm(g) <= eps:
#         break
#
#     # Step 3: steepest descent direction
#     d = -g
#
#     # Step 4: exact line search
#     lam = exact_line_search(x, d)
#
#     # Step 5: update
#     x = x + lam * d
#
#     cpu = time.time() - start
#     print(f"{k+1:<2}   {f(x):<18.10f}   {cpu:.6f}")
#
# print("\nFinal f(x) =", f(x))
# print("Total CPU time =", time.time() - start)
#
# # ======================================================
# # Damped Newton Method (Exact Line Search)
# # ======================================================
#
# print("\n========== Damped Newton Method (Exact Line Search) ==========")
#
# x = np.ones(10) * 2
# start = time.time()
#
# print("k    f(x_k)              lambda_k    CPU(s)")
#
# for k in range(max_iter):
#     g = grad(x)
#     if np.linalg.norm(g) <= eps:
#         break
#
#     H = hess(x)
#
#     # Step 3: Newton direction
#     try:
#         # check positive definiteness
#         np.linalg.cholesky(H)
#         d = -np.linalg.solve(H, g)
#     except np.linalg.LinAlgError:
#         # fallback to steepest descent
#         d = -g
#
#     # Step 4: exact line search (damping)
#     lam = exact_line_search(x, d)
#
#     # Step 5: update
#     x = x + lam * d
#
#     cpu = time.time() - start
#     print(f"{k+1:<2}   {f(x):<18.10f}   {lam:<10.4f}   {cpu:.6f}")
#
# print("\nFinal f(x) =", f(x))
# print("Total CPU time =", time.time() - start)
import numpy as np
import time

# ======================================================
# 1. 目标函数：10维 Rosenbrock
# f(x) = sum_{i=1}^9 [(1-x_i)^2 + 100(x_{i+1}-x_i^2)^2]
# ======================================================
def f(x):
    return sum((1 - x[i])**2 + 100*(x[i+1] - x[i]**2)**2 for i in range(9))


# ======================================================
# 2. 梯度
# ======================================================
def grad(x):
    g = np.zeros_like(x)

    g[0] = -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2)

    for i in range(1, 9):
        g[i] = (-2*(1 - x[i])
                - 400*x[i]*(x[i+1] - x[i]**2)
                + 200*(x[i] - x[i-1]**2))

    g[9] = 200*(x[9] - x[8]**2)
    return g


# ======================================================
# 3. Hessian
# ======================================================
def hess(x):
    H = np.zeros((10, 10))

    for i in range(9):
        H[i, i] = 2 - 400*(x[i+1] - 3*x[i]**2)
        H[i, i+1] = -400*x[i]
        H[i+1, i] = -400*x[i]

    H[9, 9] = 200
    return H


# ======================================================
# 4. Armijo 回溯线搜索
# ======================================================
def armijo_line_search(x, d, g, c=1e-4, beta=0.5):
    lam = 1.0
    fx = f(x)
    gd = np.dot(g, d)

    while f(x + lam * d) > fx + c * lam * gd:
        lam *= beta

    return lam


# ======================================================
# 5. 最速下降法（Armijo）
# ======================================================
def steepest_descent(x0, tol=1e-6, max_iter=10):
    x = x0.copy()

    print("========== Steepest Descent Method (Armijo Line Search) ==========")
    print("k    f(x_k)              lambda_k    CPU(s)")

    start_time = time.time()

    for k in range(1, max_iter + 1):
        g = grad(x)
        d = -g
        lam = armijo_line_search(x, d, g)
        x = x + lam * d

        cpu_time = time.time() - start_time
        print(f"{k:<4d} {f(x):<20.10f} {lam:<10.4f} {cpu_time:.6f}")

    print("\nFinal f(x) =", f(x))
    print("Total CPU time =", time.time() - start_time)
    return x


# ======================================================
# 6. 阻尼牛顿法（Armijo）
# ======================================================
def damped_newton(x0, tol=1e-6, max_iter=10):
    x = x0.copy()

    print("\n========== Damped Newton Method (Armijo Line Search) ==========")
    print("k    f(x_k)              lambda_k    CPU(s)")

    start_time = time.time()

    for k in range(1, max_iter + 1):
        g = grad(x)
        H = hess(x)
        d = -np.linalg.solve(H, g)
        lam = armijo_line_search(x, d, g)
        x = x + lam * d

        cpu_time = time.time() - start_time
        print(f"{k:<4d} {f(x):<20.10f} {lam:<10.4f} {cpu_time:.6f}")

    print("\nFinal f(x) =", f(x))
    print("Total CPU time =", time.time() - start_time)
    return x


# ======================================================
# 7. 主程序（初始点 x0 = (2,2,...,2)）
# ======================================================
if __name__ == "__main__":
    x0 = np.ones(10) * 2   # ★ 作业要求的初始点

    steepest_descent(x0)
    damped_newton(x0)
