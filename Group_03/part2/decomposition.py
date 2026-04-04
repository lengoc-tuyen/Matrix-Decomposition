from utils import *
from diagonalization import *
import math

def sort_singular_values(sigma: list[float], v: list[list[float]]):
    v_cols = [get_column(v, i) for i in range(len(sigma))]
    combined = sorted(zip(sigma, v_cols), key=lambda x: x[0], reverse=True)
    sigma_new, v_new = zip(*combined)
    return list(sigma_new), list(v_new)

def svd_decomposition(A: list[list[float]]) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    W = multiply_matrix(transpose(A), A)
    eigenvalues, v = jacobi_eigen(W)
    sigma = list(map(lambda x: math.sqrt(max(0,x)), eigenvalues))
    sigma, v = sort_singular_values(sigma, v)
    l = len(sigma)
    U_cols = [[j / sigma[i] for j in (multiply_mat_vec(A, v[i]))] for i in range(len(sigma)) if sigma[i] > 1e-9]
    U = transpose(U_cols)
    m = len(A)
    n = len(A[0])
    sigma_mat = create_zero_matrix(m,n)
    for i in range(len(U_cols)):
        if i < m and i < n:
            sigma_mat[i][i] = sigma[i]
    return U, sigma_mat, v
