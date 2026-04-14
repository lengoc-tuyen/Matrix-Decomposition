import numpy as np
import random
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "part1"))
sys.path.insert(0, str(ROOT / "part2"))

from gaussian import gaussian_eliminate
from determinant import determinant
from inverse import inverse
from rank_basis import rank_and_basis
from diagonalization import diagonalize, jacobi_eigen
from decomposition import svd_decomposition


# ============================================================
# CÁC HÀM KIỂM THỬ (VERIFY)
# ============================================================

def verify_solve(A, b, x_custom):
    if isinstance(x_custom, str) or (isinstance(x_custom, list) and isinstance(x_custom[0], str)):
        print("   >> Nghiệm x: [Hệ vô nghiệm/vô số nghiệm - Không so sánh mảng]")
        return True
    A_np, b_np, x_np = np.array(A, dtype=float), np.array(b, dtype=float), np.array(x_custom, dtype=float)
    return np.allclose(A_np @ x_np, b_np)

def verify_determinant(A, det_custom):
    A_np = np.array(A, dtype=float)
    return np.isclose(det_custom, np.linalg.det(A_np))

def verify_inverse(A, inv_custom):
    if isinstance(inv_custom, str):
        print("   >> Nghịch đảo: [Ma trận suy biến - Không so sánh mảng]")
        return True
    A_np, inv_np = np.array(A, dtype=float), np.array(inv_custom, dtype=float)
    return np.allclose(A_np @ inv_np, np.eye(A_np.shape[0]))

def verify_rank(A, rank_custom):
    A_np = np.array(A, dtype=float)
    return rank_custom == np.linalg.matrix_rank(A_np)

def verify_eigen(A, eigenvals_custom, eigenvecs_custom):
    """Kiểm tra A·v = λ·v cho từng cặp (λ, v)."""
    if eigenvals_custom is None or eigenvecs_custom is None:
        return False
    A_np = np.array(A, dtype=float)
    eigenvecs_np = np.array(eigenvecs_custom, dtype=float)
    for i, lam in enumerate(eigenvals_custom):
        v = eigenvecs_np[:, i]
        if not np.allclose(A_np @ v, lam * v, atol=1e-6):
            return False
    return True

def verify_eigenvecs_orthonormal(eigenvecs_custom):
    """Kiểm tra tính trực chuẩn của tập vector riêng: VᵀV ≈ I."""
    V_np = np.array(eigenvecs_custom, dtype=float)
    n = V_np.shape[1]
    return np.allclose(V_np.T @ V_np, np.eye(n), atol=1e-6)

def verify_diagonalize(A, P, D, P_transpose):
    """Kiểm tra A ≈ P·D·Pᵀ."""
    A_np  = np.array(A, dtype=float)
    P_np  = np.array(P, dtype=float)
    D_np  = np.array(D, dtype=float)
    Pt_np = np.array(P_transpose, dtype=float)
    return np.allclose(A_np, P_np @ D_np @ Pt_np, atol=1e-6)

def verify_svd(A, U, sigma_mat, V):
    """Kiểm tra A ≈ U·Σ·Vᵀ."""
    A_np     = np.array(A, dtype=float)
    U_np     = np.array(U, dtype=float)
    sigma_np = np.array(sigma_mat, dtype=float)
    Vt_np    = np.array(V, dtype=float)
    return np.allclose(A_np, U_np @ sigma_np @ Vt_np, atol=1e-6)


# ============================================================
# SINH DỮ LIỆU TEST
# ============================================================

def generate_large_test(n):
    A = [[random.randint(-5, 5) for _ in range(n)] for _ in range(n)]
    b = [random.randint(-10, 10) for _ in range(n)]
    return A, b

def generate_symmetric_test(n):
    B = [[random.randint(-5, 5) for _ in range(n)] for _ in range(n)]
    return [[B[i][j] + B[j][i] for j in range(n)] for i in range(n)]


# ============================================================
# HÀM CHẠY TEST
# ============================================================

def run_quick_tests():

    # PHAN 1: Gauss / Dinh thuc / Nghich dao / Rank
    gauss_cases = [
        {"name": "He co nghiem duy nhat",
         "A": [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]], "b": [8, -11, -3]},
        {"name": "Pivot = 0",
         "A": [[0, 2, 1], [1, -2, -3], [-1, 1, 2]], "b": [-8, 0, 3]},
        {"name": "Vo so nghiem",
         "A": [[1, 1, 1], [2, 2, 2], [3, 3, 3]], "b": [3, 6, 9]},
        {"name": "Vo nghiem",
         "A": [[1, 1], [1, 1]], "b": [1, 2]},
        {"name": "Suy bien",
         "A": [[1, 2], [2, 4]], "b": [3, 6]},
        {"name": "Ma tran don vi",
         "A": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "b": [5, -3, 2]},
        {"name": "Zero matrix",
         "A": [[0, 0], [0, 0]], "b": [0, 0]},
        {"name": "Gan suy bien",
         "A": [[1, 1, 1], [1, 1.0000001, 1], [1, 1, 1.0000001]],
         "b": [3, 3.0000001, 3.0000001]},
        {"name": "4x4",
         "A": [[1,2,3,4],[2,5,2,1],[3,2,6,2],[4,1,2,7]], "b": [10,8,9,11]},
        {"name": "Rank < n",
         "A": [[1,2,3],[2,4,6],[0,0,0]], "b": [6,12,0]},
    ]
    for size in [30, 50, 80]:
        A, b = generate_large_test(size)
        gauss_cases.append({"name": f"Large random {size}x{size}",
                             "A": A, "b": b, "large": True})

    # PHAN 2: Cheo hoa & SVD
    diag_svd_cases = [
        {"name": "2x2 don gian",             "A": [[4, 1], [1, 3]]},
        {"name": "3x3 doi xung",             "A": [[4, 2, 0], [2, 3, 1], [0, 1, 2]]},
        {"name": "Ma tran don vi 3x3",       "A": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
        {"name": "Ma tran khong kha nghich", "A": [[1, 2, 3], [2, 4, 6], [3, 6, 9]]},
        {"name": "4x4 doi xung",             "A": [[5,2,1,0],[2,4,2,1],[1,2,5,2],[0,1,2,4]]},
    ]
    for size in [10, 20]:
        diag_svd_cases.append({"name": f"Large symmetric {size}x{size}",
                                "A": generate_symmetric_test(size), "large": True})

    # PHAN 3: Tri rieng & Vector rieng
    eigen_cases = [
        {"name": "Ma tran duong cheo (lam hien nhien)",
         "A": [[3, 0, 0], [0, 1, 0], [0, 0, 5]]},
        {"name": "2x2 don gian",
         "A": [[4, 1], [1, 3]]},
        {"name": "3x3 doi xung",
         "A": [[6, 2, 1], [2, 3, 1], [1, 1, 1]]},
        {"name": "Co tri rieng am",
         "A": [[-2, 1], [1, -3]]},
        {"name": "Ma tran don vi (moi lam = 1)",
         "A": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
        {"name": "Ma tran suy bien (co lam = 0)",
         "A": [[1, 2, 3], [2, 4, 6], [3, 6, 9]]},
        {"name": "4x4 doi xung",
         "A": [[5, 2, 1, 0], [2, 4, 2, 1], [1, 2, 5, 2], [0, 1, 2, 4]]},
        {"name": "Large symmetric 10x10",
         "A": generate_symmetric_test(10), "large": True},
        {"name": "Large symmetric 20x20",
         "A": generate_symmetric_test(20), "large": True},
    ]

    # RUN PHAN 1
    print("=" * 60)
    print("PHAN 1: GAUSS / DINH THUC / NGHICH DAO / RANK")
    print("=" * 60)
    for idx, tc in enumerate(gauss_cases):
        name = tc["name"]
        A, b = tc["A"], tc["b"]
        is_large = tc.get("large", False)
        print(f"\n Test case {idx+1}: {name}")
        if not is_large:
            print(f"   A = {A}")
            print(f"   b = {b}")
        else:
            print(f"   (Ma tran lon {len(A)}x{len(A[0])} - khong hien thi)")
        try:
            my_det  = determinant(A)
            my_rank = rank_and_basis(A)[0]
            _, my_x, _ = gaussian_eliminate(A, b)
            my_inv  = inverse(A)
            errors = []
            if not verify_solve(A, b, my_x):     errors.append("Sai nghiem Ax = b")
            if not verify_determinant(A, my_det): errors.append("Sai dinh thuc")
            if not verify_inverse(A, my_inv):     errors.append("Sai nghich dao")
            if not verify_rank(A, my_rank):       errors.append("Sai rank")
            print("   OK Passed" if not errors else "   FAIL: " + ", ".join(errors))
        except Exception as e:
            print(f"   CRASH: {e}")

    # RUN PHAN 2
    print("\n" + "=" * 60)
    print("PHAN 2: CHEO HOA (diagonalize) & PHAN RA SVD")
    print("=" * 60)
    for idx, tc in enumerate(diag_svd_cases):
        name = tc["name"]
        A = tc["A"]
        is_large = tc.get("large", False)
        print(f"\n Test case {idx+1}: {name}")
        if not is_large:
            print(f"   A = {A}")
        else:
            print(f"   (Ma tran lon {len(A)}x{len(A[0])} - khong hien thi)")
        errors = []
        try:
            P, D, Pt = diagonalize(A)
            if not verify_diagonalize(A, P, D, Pt): errors.append("Cheo hoa sai")
        except Exception as e:
            errors.append(f"Cheo hoa CRASH: {e}")
        try:
            U, sigma_mat, V = svd_decomposition(A)
            if not verify_svd(A, U, sigma_mat, V): errors.append("SVD sai")
        except Exception as e:
            errors.append(f"SVD CRASH: {e}")
        print("   OK Passed" if not errors else "   FAIL: " + ", ".join(errors))

    # RUN PHAN 3
    print("\n" + "=" * 60)
    print("PHAN 3: TRI RIENG & VECTOR RIENG (jacobi_eigen)")
    print("=" * 60)
    for idx, tc in enumerate(eigen_cases):
        name = tc["name"]
        A = tc["A"]
        is_large = tc.get("large", False)
        print(f"\n Test case {idx+1}: {name}")
        if not is_large:
            print(f"   A = {A}")
        else:
            print(f"   (Ma tran lon {len(A)}x{len(A[0])} - khong hien thi)")
        try:
            eigenvals, eigenvecs_mat = jacobi_eigen(A)
            if not is_large:
                vals_str = ", ".join(f"{v:.4f}" for v in eigenvals)
                print(f"   lam = [{vals_str}]")
            errors = []
            if not verify_eigen(A, eigenvals, eigenvecs_mat):
                errors.append("Sai dieu kien A.v = lam.v")
            if not verify_eigenvecs_orthonormal(eigenvecs_mat):
                errors.append("Vector rieng khong truc chuan (Vt.V != I)")
            print("   OK Passed" if not errors else "   FAIL: " + ", ".join(errors))
        except Exception as e:
            print(f"   CRASH: {e}")

    print("\n" + "=" * 60)
    print("KET THUC")
    print("=" * 60)


run_quick_tests()
