from .utils import get_column, multiply_mat_vec, multiply_matrix, transpose
from utils import get_column, multiply_mat_vec, multiply_matrix, transpose
from .diagonalization import diagonalize, jacobi_eigen
from diagonalization import diagonalize, jacobi_eigen
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
    U = [[j /sigma[i] for j in (multiply_mat_vec(A, v[i]))] for i in range(l) if sigma[i] > 1e-9]
    U = transpose(U)
    return U, sigma, v
