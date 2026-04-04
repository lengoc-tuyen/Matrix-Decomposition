from gaussian import gaussian_eliminate


def determinant(A):
    n = len(A)
    for row in A:
        if len(row) != n:
            raise ValueError("Ma trận phải là ma trận vuông.")
    b_zero = [0.0] * n
    U, _, swaps = gaussian_eliminate(A, b_zero)


    det_val = (-1) ** swaps
    for i in range(n):
        det_val *= U[i][i]


    return det_val
