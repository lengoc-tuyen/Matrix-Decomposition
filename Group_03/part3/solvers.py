import numpy as np
import random
import sys
from pathlib import Path

# Add part1 & part2 to path
sys.path.insert(0, str(Path(__file__).parent.parent / "part1"))
sys.path.insert(0, str(Path(__file__).parent.parent / "part2"))

from determinant import determinant
from gaussian import gaussian_eliminate
from rank_basis import rank_and_basis
from inverse import inverse
from diagonalization import diagonalize
from decomposition import svd_decomposition

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
    if eigenvals_custom is None or eigenvecs_custom is None:
        return False
        
    A_np = np.array(A, dtype=float)
    eigenvecs_np = np.array(eigenvecs_custom, dtype=float)
    for i, lam in enumerate(eigenvals_custom):
        v = eigenvecs_np[:, i]
        if not np.allclose(A_np @ v, lam * v):
            return False
    return True


def verify_diagonalize(A, P, D, P_transpose):
    A_np  = np.array(A, dtype=float)
    P_np  = np.array(P, dtype=float)
    D_np  = np.array(D, dtype=float)
    Pt_np = np.array(P_transpose, dtype=float)
    reconstructed = P_np @ D_np @ Pt_np
    return np.allclose(A_np, reconstructed, atol=1e-6)


def verify_svd(A, U, sigma_mat, V):
    A_np     = np.array(A, dtype=float)
    U_np     = np.array(U, dtype=float)
    sigma_np = np.array(sigma_mat, dtype=float)
    Vt_np    = np.array(V, dtype=float)
    reconstructed = U_np @ sigma_np @ Vt_np
    return np.allclose(A_np, reconstructed, atol=1e-6)


def generate_large_test(n):
    """Sinh test case lớn n x n"""
    A = [[random.randint(-5, 5) for _ in range(n)] for _ in range(n)]
    b = [random.randint(-10, 10) for _ in range(n)]
    return A, b


def generate_symmetric_test(n):
    """Sinh ma trận đối xứng ngẫu nhiên n x n (dùng cho chéo hóa & SVD)."""
    B = [[random.randint(-5, 5) for _ in range(n)] for _ in range(n)]
    return [[B[i][j] + B[j][i] for j in range(n)] for i in range(n)]


def run_quick_tests():
    """Chạy test: Gauss + Định thức + Nghịch đảo + Rank + Chéo hóa + SVD."""

    gauss_cases = [
        {
            "name": "Hệ có nghiệm duy nhất",
            "A": [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]],
            "b": [8, -11, -3]
        },
        {
            "name": "Pivot = 0",
            "A": [[0, 2, 1], [1, -2, -3], [-1, 1, 2]],
            "b": [-8, 0, 3]
        },
        {
            "name": "Vô số nghiệm",
            "A": [[1, 1, 1], [2, 2, 2], [3, 3, 3]],
            "b": [3, 6, 9]
        },
        {
            "name": "Vô nghiệm",
            "A": [[1, 1], [1, 1]],
            "b": [1, 2]
        },
        {
            "name": "Suy biến",
            "A": [[1, 2], [2, 4]],
            "b": [3, 6]
        },
        {
            "name": "Ma trận đơn vị",
            "A": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "b": [5, -3, 2]
        },
        {
            "name": "Zero matrix",
            "A": [[0, 0], [0, 0]],
            "b": [0, 0]
        },
        {
            "name": "Gần suy biến",
            "A": [[1, 1, 1], [1, 1.0000001, 1], [1, 1, 1.0000001]],
            "b": [3, 3.0000001, 3.0000001]
        },
        {
            "name": "4x4",
            "A": [[1,2,3,4],[2,5,2,1],[3,2,6,2],[4,1,2,7]],
            "b": [10,8,9,11]
        },
        {
            "name": "Rank < n",
            "A": [[1,2,3],[2,4,6],[0,0,0]],
            "b": [6,12,0]
        },
    ]

    # Thêm test lớn cho Gauss
    for size in [30, 50, 80]:
        A, b = generate_large_test(size)
        gauss_cases.append({
            "name": f"Large random {size}x{size}",
            "A": A, "b": b, "large": True
        })

    # ── Các test case chéo hóa & SVD (ma trận đối xứng) ──
    diag_svd_cases = [
        {
            "name": "Chéo hóa / SVD – 2x2 đơn giản",
            "A": [[4, 1], [1, 3]]
        },
        {
            "name": "Chéo hóa / SVD – 3x3 đối xứng",
            "A": [[4, 2, 0], [2, 3, 1], [0, 1, 2]]
        },
        {
            "name": "Chéo hóa / SVD – Ma trận đơn vị 3x3",
            "A": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        },
        {
            "name": "Chéo hóa / SVD – Ma trận không khả nghịch",
            "A": [[1, 2, 3], [2, 4, 6], [3, 6, 9]]
        },
        {
            "name": "Chéo hóa / SVD – 4x4 đối xứng",
            "A": [[5,2,1,0],[2,4,2,1],[1,2,5,2],[0,1,2,4]]
        },
    ]

    # Thêm test lớn cho chéo hóa & SVD
    for size in [10, 20]:
        diag_svd_cases.append({
            "name": f"Chéo hóa / SVD – Large symmetric {size}x{size}",
            "A": generate_symmetric_test(size),
            "large": True
        })

    # ════════════════════════════════════════════════════════
    print("=" * 60)
    print("PHẦN 1: GAUSS / ĐỊNH THỨC / NGHỊCH ĐẢO / RANK")
    print("=" * 60)

    for idx, tc in enumerate(gauss_cases):
        print(f"\n Test case {idx+1}: {tc['name']}")
        A, b = tc["A"], tc["b"]
        if not tc.get("large", False):
            print(f"   A = {A}")
            print(f"   b = {b}")
        else:
            print(f"   (Ma trận lớn {len(A)}x{len(A[0])} – không hiển thị)")

        try:
            my_det  = determinant(A)
            my_rank = rank_and_basis(A)[0]
            _, my_x, _ = gaussian_eliminate(A, b)
            my_inv  = inverse(A)

            errors = []
            if not verify_solve(A, b, my_x):       errors.append("Sai nghiệm Ax = b")
            if not verify_determinant(A, my_det):   errors.append("Sai định thức")
            if not verify_inverse(A, my_inv):       errors.append("Sai nghịch đảo")
            if not verify_rank(A, my_rank):         errors.append("Sai rank")

            if not errors:
                print("   Passed")
            else:
                print("   Failed:")
                for err in errors: print(f"      - {err}")
        except Exception as e:
            print(f"   CRASH: {e}")

    # ════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PHẦN 2: CHÉO HÓA (diagonalize) & PHÂN RÃ SVD")
    print("=" * 60)

    for idx, tc in enumerate(diag_svd_cases):
        print(f"\n Test case {idx+1}: {tc['name']}")
        A = tc["A"]
        if not tc.get("large", False):
            print(f"   A = {A}")
        else:
            print(f"   (Ma trận lớn {len(A)}x{len(A[0])} – không hiển thị)")

        errors = []

        # --- Kiểm tra chéo hóa: A ≈ P * D * P^T ---
        try:
            P, D, Pt = diagonalize(A)
            if not verify_diagonalize(A, P, D, Pt):
                errors.append("Chéo hóa sai: A ≠ P·D·Pᵀ")
        except Exception as e:
            errors.append(f"Chéo hóa CRASH: {e}")

        # --- Kiểm tra SVD: A ≈ U * Σ * Vᵀ ---
        try:
            U, sigma_mat, V = svd_decomposition(A)
            if not verify_svd(A, U, sigma_mat, V):
                errors.append("SVD sai: A ≠ U·Σ·Vᵀ")
        except Exception as e:
            errors.append(f"SVD CRASH: {e}")

        if not errors:
            print("   Passed")
        else:
            print("   Failed:")
            for err in errors: print(f"      - {err}")

    print("\n" + "=" * 60)
    print("KẾT THÚC")
    print("=" * 60)

run_quick_tests()