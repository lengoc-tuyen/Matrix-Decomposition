from utils import *
from diagonalization import *
import math

def sort_singular_values(sigma: list[float], v: list[list[float]]):
    v_cols = [get_column(v, i) for i in range(len(sigma))]
    combined = sorted(zip(sigma, v_cols), key=lambda x: x[0], reverse=True)
    sigma_new, v_new = zip(*combined)
    return list(sigma_new), list(v_new)

def svd_decomposition(A: list[list[float]]) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    m = len(A)
    n = len(A[0])
    W = multiply_matrix(transpose(A), A)
    eigenvalues, v = jacobi_eigen(W)
    
    sigma = [math.sqrt(max(0, x)) for x in eigenvalues]
    sigma, v = sort_singular_values(sigma, v)
    
    U_cols = []
    for i in range(len(sigma)):
        if sigma[i] > 1e-9:
            avi = multiply_mat_vec(A, v[i])
            ui = [val / sigma[i] for val in avi]
            U_cols.append(ui)
    
    if len(U_cols) < m:
        I = create_identity_matrix(m)
        for i in range(m):
            candidate_col = [I[row][i] for row in range(m)]
            U_cols.append(candidate_col)
            U_cols = gram_schmidt(U_cols)
            
            if len(U_cols) == m:
                break
    
    U = transpose(U_cols)
    
    sigma_mat = create_zero_matrix(m, n)
    for i in range(min(m, n)):
        if i < len(sigma):
            sigma_mat[i][i] = sigma[i]
            
    return U, sigma_mat, v