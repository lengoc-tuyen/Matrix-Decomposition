from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "part1"))
sys.path.insert(0, str(ROOT / "part3"))

from gaussian import back_substitution, gaussian_eliminate
from determinant import determinant
from inverse import inverse
from rank_basis import rank_and_basis

try:
    from solvers import verify_solution
    SOLVER_IMPORT_ERROR = None
except Exception as exc:
    verify_solution = None
    SOLVER_IMPORT_ERROR = exc


def print_matrix(M, title=None, digits=4):
    if title:
        print(title)
    for row in M:
        print("  [" + ", ".join(f"{v:.{digits}f}" for v in row) + "]")


def demo_gaussian_and_back_substitution():
    print("\n=== Demo 1: Gaussian Elimination + Back Substitution ===")

    A = [
        [2.0, 1.0, -1.0],
        [-3.0, -1.0, 2.0],
        [-2.0, 1.0, 2.0],
    ]
    b = [8.0, -11.0, -3.0]

    U, x, swaps = gaussian_eliminate(A, b)
    print_matrix(U, "U after elimination:")
    print(f"So lan hoan vi dong: {swaps}")
    print(f"Nghiem he Ax=b: {x}")

    U2 = [
        [1.0, 2.0, 3.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
    ]
    c2 = [4.0, 2.0, 0.0]
    x2 = back_substitution(U2, c2)
    print("He bac thang co vo so nghiem:")
    print(x2)

    if verify_solution is not None and isinstance(x, list):
        err = verify_solution(A, x, b)
        print(f"Sai so kiem tra ||Ax-b||/||b||: {err:.6e}")
    elif SOLVER_IMPORT_ERROR is not None:
        print("Bo qua verify_solution vi khong import duoc part3/solvers.py")
        print(f"Ly do: {SOLVER_IMPORT_ERROR}")


def demo_determinant_and_inverse():
    print("\n=== Demo 2: Determinant + Inverse ===")

    A = [
        [4.0, 7.0],
        [2.0, 6.0],
    ]

    detA = determinant(A)
    invA = inverse(A)

    print(f"det(A) = {detA}")
    if isinstance(invA, str):
        print(invA)
    else:
        print_matrix(invA, "A^-1:")


def demo_rank_and_basis():
    print("\n=== Demo 3: Rank and Basis ===")

    A = [
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 4.0, 6.0, 8.0],
        [1.0, 1.0, 1.0, 1.0],
    ]

    rank, bases = rank_and_basis(A)
    print(f"rank(A) = {rank}")
    print(f"column_space basis: {bases['column_space']}")
    print(f"row_space basis: {bases['row_space']}")
    print(f"null_space basis: {bases['null_space']}")


def main():
    print("Run matrix decomposition project demo")
    demo_gaussian_and_back_substitution()
    demo_determinant_and_inverse()
    demo_rank_and_basis()


if __name__ == "__main__":
    main()
