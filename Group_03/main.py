from pathlib import Path
import sys
import io
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "part1"))
sys.path.insert(0, str(ROOT / "part2"))
sys.path.insert(0, str(ROOT / "part3"))

from part1.gaussian import back_substitution, gaussian_eliminate  
from part1.determinant import determinant
from part1.inverse import inverse
from part1.rank_basis import rank_and_basis
from part2.utils import multiply_matrix, transpose
from part2.diagonalization import diagonalize
from part2.decomposition import svd_decomposition

import numpy as np

def verify_solution(A, x, b):
    """Kiểm tra sai số tương đối của phương trình Ax = b"""
    A_np = np.array(A, dtype=float)
    x_np = np.array(x, dtype=float)
    b_np = np.array(b, dtype=float)
    
    numerator = np.linalg.norm(np.dot(A_np, x_np) - b_np, 2)
    denominator = np.linalg.norm(b_np, 2)
    
    if denominator == 0:
        return float('inf')
    return numerator / denominator
def print_matrix(matrix, title=None, digits=4):
    if title:
        print(title)
    for row in matrix:
        print("  [" + ", ".join(f"{value:.{digits}f}" for value in row) + "]")


def print_vector(vector, title=None, digits=4):
    if title:
        print(title)
    print("  [" + ", ".join(f"{value:.{digits}f}" for value in vector) + "]")


def pretty_separator(title):
    print("\n" + "=" * 10 + f" {title} " + "=" * 10)


def demo_part1_gaussian():
    pretty_separator("PART 1 - GAUSSIAN")

    A = [
        [2.0, 1.0, -1.0],
        [-3.0, -1.0, 2.0],
        [-2.0, 1.0, 2.0],
    ]
    b = [8.0, -11.0, -3.0]

    U, x, swaps = gaussian_eliminate(A, b)
    print_matrix(U, "Upper triangular matrix U:")
    print(f"Number of row swaps: {swaps}")
    print_vector(x, "Solution x:")

    if isinstance(x, list):
        try:
            err = verify_solution(A, x, b)
            print(f"Verification error ||Ax-b||/||b|| = {err:.6e}")
        except Exception as e:
            print(f"Verification could not be processed: {e}")

    U2 = [
        [1.0, 2.0, 3.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
    ]
    c2 = [4.0, 2.0, 0.0]
    print("General solution case:")
    print(back_substitution(U2, c2))

    U3 = [
        [1.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]
    c3 = [1.0, 1.0, 0.0]
    print("Inconsistent system case:")
    print(back_substitution(U3, c3))


def demo_part1_determinant_inverse_rank():
    pretty_separator("PART 1 - DETERMINANT / INVERSE / RANK")

    A = [
        [4.0, 7.0],
        [2.0, 6.0],
    ]
    det_a = determinant(A)
    print(f"det(A) = {det_a}")

    inv_a = inverse(A)
    if isinstance(inv_a, str):
        print(inv_a)
    else:
        print_matrix(inv_a, "A inverse:")
        print_matrix(multiply_matrix(A, inv_a), "A * A^-1:")

    B = [
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 4.0, 6.0, 8.0],
        [1.0, 1.0, 1.0, 1.0],
    ]
    rank, bases = rank_and_basis(B)
    print(f"rank(B) = {rank}")
    print(f"column_space basis = {bases['column_space']}")
    print(f"row_space basis = {bases['row_space']}")
    print(f"null_space basis = {bases['null_space']}")


def demo_part2_diagonalization():
    pretty_separator("PART 2 - DIAGONALIZATION")

    A = [
        [4.0, 1.0],
        [1.0, 3.0],
    ]

    P, D, P_inv = diagonalize(A)
    print_matrix(P, "P:")
    print_matrix(D, "D:")
    print_matrix(P_inv, "P_inv:")
    print_matrix(multiply_matrix(multiply_matrix(P, D), P_inv), "P * D * P_inv:")


def demo_part2_svd():
    pretty_separator("PART 2 - SVD")

    A = [
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 3.0],
    ]

    U, Sigma, Vt = svd_decomposition(A)
    print_matrix(U, "U:")
    print_matrix(Sigma, "Sigma:")
    print_matrix(Vt, "V^T:")
    print_matrix(multiply_matrix(multiply_matrix(U, Sigma), Vt), "U * Sigma * V^T:")


def demo_transpose():
    pretty_separator("UTILS CHECK")
    C = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ]
    print_matrix(transpose(C), "transpose(C):")


def main():
    print("Run matrix decomposition full test suite")
    demo_transpose()
    demo_part1_gaussian()
    demo_part1_determinant_inverse_rank()
    demo_part2_diagonalization()
    demo_part2_svd()


if __name__ == "__main__":
    main()
