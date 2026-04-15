import numpy as np
import sys
import os

# =========================================================================== #
#  CẤU HÌNH ĐƯỜNG DẪN GỐC ĐỂ IMPORT LIÊN THƯ MỤC
# =========================================================================== #
current_dir = os.path.dirname(os.path.abspath(__file__)) # Thư mục hiện tại (part3)
root_dir = os.path.abspath(os.path.join(current_dir, "..")) # Thư mục gốc

# 1. Thêm thư mục gốc
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# 2. THÊM TRỰC TIẾP THƯ MỤC PART 1 VÀ PART 2 VÀO HỆ THỐNG
# (Chiêu thức này giúp các file bên trong part1/part2 tự động nhìn thấy nhau)
part1_dir = os.path.join(root_dir, "part1")
if part1_dir not in sys.path:
    sys.path.insert(0, part1_dir)

part2_dir = os.path.join(root_dir, "part2")
if part2_dir not in sys.path:
    sys.path.insert(0, part2_dir)

# =========================================================================== #
#  IMPORT TỪ PART 1 VÀ PART 2
# =========================================================================== #
try:
    from part1.gaussian import gaussian_eliminate, back_substitution
    _GAUSS_AVAILABLE = True
    print("[solvers.py] Đã nạp thành công Khử Gauss từ part1/gaussian.py")
except ImportError as e:
    _GAUSS_AVAILABLE = False
    print(f"[solvers.py] CẢNH BÁO: Không nạp được part1/gaussian.py — Lỗi: {e}")

try:
    from part2.decomposition import svd_decomposition
    _SVD_AVAILABLE = True
    print("[solvers.py] Đã nạp thành công SVD từ part2/decomposition.py")
except ImportError as e:
    _SVD_AVAILABLE = False
    print(f"[solvers.py] CẢNH BÁO: Không nạp được part2/decomposition.py — Lỗi: {e}")

# ... (Giữ nguyên các hàm solve_gauss, solve_svd bên dưới) ...


# =========================================================================== #
#  PHẦN 1 — Solver: Gauss với Partial Pivoting
# =========================================================================== #
def solve_gauss(A, b):
    """
    Giải hệ Ax = b bằng khử Gauss với Partial Pivoting (Phần 1).
    """
    if not _GAUSS_AVAILABLE:
        raise ImportError("part1/gaussian.py chưa được nạp. Vui lòng kiểm tra lại cấu trúc file.")

    _, x, swaps = gaussian_eliminate(A, b)
    return x, {"swaps": swaps, "method": "Gauss–Partial Pivoting"}


# =========================================================================== #
#  PHẦN 2 — Solver: SVD (A = U Σ Vᵀ  →  x = V Σ⁺ Uᵀ b)
# =========================================================================== #
def solve_svd(A, b):
    """
    Giải hệ Ax = b (hoặc bài toán least-squares) bằng phân rã SVD.
    """
    if not _SVD_AVAILABLE:
        raise ImportError("part2/decomposition.py chưa được nạp. Vui lòng kiểm tra lại cấu trúc file.")

    m = len(A)
    n = len(A[0])
    epsilon = 1e-10

    U_raw, sigma_mat, Vt_raw = svd_decomposition(A)

    U   = [[float(U_raw[i][j])    for j in range(len(U_raw[0]))]    for i in range(len(U_raw))]
    Vt  = [[float(Vt_raw[i][j])   for j in range(len(Vt_raw[0]))]   for i in range(len(Vt_raw))]
    
    p = min(m, n)
    sigmas = [sigma_mat[i][i] for i in range(p)]
   
    Ut_b = [sum(U[k][i] * float(b[k]) for k in range(m)) for i in range(len(U[0]))]
   
    sigma_plus_Utb = []
    rank = 0
    for i in range(p):
        if abs(sigmas[i]) > epsilon:
            sigma_plus_Utb.append(Ut_b[i] / sigmas[i])
            rank += 1
        else:
            sigma_plus_Utb.append(0.0)
   
    x = []
    for j in range(n):
        val = sum(Vt[i][j] * sigma_plus_Utb[i] for i in range(p))
        x.append(val)

    return x, {
        "singular_values": sigmas,
        "rank": rank,
        "method": "SVD (pseudo-inverse)"
    }

def is_strictly_diag_dominant(A):
    """
    Kiểm tra ma trận A có chéo trội chặt hàng không.

    Điều kiện:  |a_ii| > Σ_{j≠i} |a_ij|  với mọi i.

    Tham số
    -------
    A : list[list[float]]

    Trả về
    ------
    (bool, list[float])
        - True/False  : có / không chéo trội
        - margins     : |a_ii| - Σ_{j≠i}|a_ij|  cho từng hàng i
                        (số dương ↔ hàng đó thỏa, âm ↔ không thỏa)
    """
    n = len(A)
    margins = []
    for i in range(n):
        diag   = abs(A[i][i])
        off    = sum(abs(A[i][j]) for j in range(n) if j != i)
        margins.append(diag - off)
    dominant = all(m > 0 for m in margins)
    return dominant, margins


def solve_gauss_seidel(A, b,
                       x0=None,
                       tol=1e-10,
                       max_iter=10_000,
                       force=False):
    """
    Giải hệ Ax = b bằng phương pháp lặp Gauss–Seidel.

    Công thức lặp (theo từng thành phần):
        x_i^(k+1) = (1/a_ii) * [ b_i
                                  - Σ_{j<i}  a_ij * x_j^(k+1)   ← đã cập nhật
                                  - Σ_{j>i}  a_ij * x_j^(k)  ]  ← chưa cập nhật

    Điều kiện hội tụ đảm bảo: A chéo trội chặt hàng.
    Nếu không thỏa mà force=False thì hàm phát cảnh báo nhưng vẫn chạy.

    Tham số
    -------
    A        : list[list[float]]  — Ma trận vuông n × n
    b        : list[float]        — Vế phải
    x0       : list[float] | None — Điểm khởi đầu (mặc định: vector 0)
    tol      : float              — Ngưỡng hội tụ (||x^(k+1) - x^(k)||_inf)
    max_iter : int                — Số vòng lặp tối đa
    force    : bool               — True = chạy dù không chéo trội

    Trả về
    ------
    x        : list[float]  — Nghiệm xấp xỉ
    info     : dict
        'converged'        : bool
        'iterations'       : int
        'residual_norm'    : float  (||Ax - b||_2)
        'diag_dominant'    : bool
        'method'           : str
    """
    n = len(A)


    if n != len(A[0]):
        raise ValueError("Gauss–Seidel chỉ áp dụng cho ma trận vuông.")

   
    for i in range(n):
        if abs(A[i][i]) < 1e-14:
            raise ValueError(
                f"Phần tử chéo A[{i}][{i}] = 0. "
                "Hãy hoán đổi hàng để đưa phần tử khác 0 về đường chéo."
            )

   
    dominant, margins = is_strictly_diag_dominant(A)
    if not dominant and not force:
        print(
            "[Gauss–Seidel] CẢNH BÁO: Ma trận KHÔNG chéo trội chặt hàng — "
            "hội tụ KHÔNG được đảm bảo.\n"
            "  margins = " + str([round(m, 6) for m in margins]) + "\n"
            "  Truyền force=True để bỏ qua cảnh báo này."
        )

  
    x = [float(v) for v in x0] if x0 is not None else [0.0] * n

    converged = False
    it = 0

    
    for it in range(1, max_iter + 1):
        x_old = x[:]

        for i in range(n):
            s1 = sum(A[i][j] * x[j]     for j in range(i))
            s2 = sum(A[i][j] * x_old[j] for j in range(i + 1, n))
            x[i] = (b[i] - s1 - s2) / A[i][i]
        diff = max(abs(x[j] - x_old[j]) for j in range(n))
        if diff < tol:
            converged = True
            break

    residual = [
        sum(A[i][j] * x[j] for j in range(n)) - b[i]
        for i in range(n)
    ]
    residual_norm = sum(r ** 2 for r in residual) ** 0.5

    return x, {
        "converged":     converged,
        "iterations":    it,
        "residual_norm": residual_norm,
        "diag_dominant": dominant,
        "method":        "Gauss–Seidel"
    }

def _residual_norm(A, x, b):
    """Tính ||Ax - b||_2. Dùng nội bộ để verify."""
    if isinstance(x, list) and isinstance(x[0], float):
        n = len(b)
        res = [sum(A[i][j] * x[j] for j in range(len(x))) - b[i] for i in range(n)]
        return sum(r ** 2 for r in res) ** 0.5
    return float("inf")


def _rel_error(x_approx, x_ref):
    """Sai số tương đối ||x_approx - x_ref||_2 / ||x_ref||_2."""
    if not isinstance(x_approx, list) or not isinstance(x_ref, list):
        return float("inf")
    diff_sq = sum((a - b) ** 2 for a, b in zip(x_approx, x_ref))
    ref_sq  = sum(v ** 2 for v in x_ref)
    if ref_sq < 1e-30:
        return diff_sq ** 0.5
    return (diff_sq / ref_sq) ** 0.5


def _print_result(label, x, info, A=None, b=None, x_ref=None):
    """In kết quả ngắn gọn."""
    status = ""
    if isinstance(x, str):
        status = f"  → {x}"
    elif isinstance(x, list) and len(x) <= 6:
        status = f"  x = [{', '.join(f'{v:.6f}' for v in x)}]"
    else:
        status = f"  x[0..2] = [{', '.join(f'{v:.6f}' for v in x[:3])}  …]"

    print(f"  [{label}]{status}")

    if info:
        method = info.get("method", "")
        print(f"    method     : {method}")
        if "iterations" in info:
            conv = "✓ hội tụ" if info["converged"] else "✗ không hội tụ"
            print(f"    iterations : {info['iterations']}  ({conv})")
        if "residual_norm" in info:
            print(f"    residual   : {info['residual_norm']:.2e}")
        if "rank" in info:
            print(f"    rank       : {info['rank']}")

    if x_ref is not None and isinstance(x, list) and isinstance(x[0], float):
        err = _rel_error(x, x_ref)
        print(f"    rel_error  : {err:.2e}")


def _run_tests():
    """
    Bộ test cases toàn diện:
      - 5 test cho solve_gauss
      - 5 test cho solve_svd
      - 5 test cho solve_gauss_seidel
      - 2 test bổ sung kiểm tra is_strictly_diag_dominant
    """
    PASS = "✓ PASS"
    FAIL = "✗ FAIL"
    tol  = 1e-6

    def check(label, x, x_ref=None, converged=None):
        """Trả về True nếu test qua."""
        # Trường hợp nghiệm đặc biệt (string)
        if isinstance(x, str):
            ok = (x_ref == "special")
            print(f"    {PASS if ok else FAIL}  [{label}]  Nghiệm đặc biệt: '{x}'")
            return ok
        if x_ref is None:
            print(f"    {PASS}  [{label}]  (không có x_ref để so sánh)")
            return True
        if not isinstance(x, list):
            print(f"    {FAIL}  [{label}]  x không phải list")
            return False
        err = _rel_error(x, x_ref)
        ok  = (err < tol)
        if converged is not None:
            ok = ok and converged
        print(f"    {PASS if ok else FAIL}  [{label}]  rel_error={err:.2e}"
              + (f"  converged={converged}" if converged is not None else ""))
        return ok

    results = {}  

    print("\n" + "=" * 65)
    print("  TEST 1/3 — solve_gauss (Gauss Partial Pivoting)")
    print("=" * 65)
    g_results = []

    if _GAUSS_AVAILABLE:
       
        A = [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]]
        b = [8.0, -11.0, -3.0]
        x, _ = solve_gauss(A, b)
        g_results.append(check("G1: 3×3 nghiệm duy nhất", x, [2.0, 3.0, -1.0]))

       
        A = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        b = [5.0, -3.0, 2.0]
        x, _ = solve_gauss(A, b)
        g_results.append(check("G2: Ma trận đơn vị", x, [5.0, -3.0, 2.0]))

       
        A = [[0, 2, 1], [1, -2, -3], [-1, 1, 2]]
        b = [-8.0, 0.0, 3.0]
        x, _ = solve_gauss(A, b)
        x_ref = np.linalg.solve(np.array(A, float), np.array(b, float)).tolist()
        g_results.append(check("G3: Pivot=0 tại (0,0)", x, x_ref))

       
        np.random.seed(42)
        A4 = np.random.randint(-5, 6, (4, 4)).tolist()
        b4 = np.random.randint(-10, 11, 4).tolist()
    
        A4[0][0] += 20
        A4_np = np.array(A4, float)
        x_ref4 = np.linalg.solve(A4_np, np.array(b4, float)).tolist()
        x, _ = solve_gauss(A4, b4)
        g_results.append(check("G4: 4×4 ngẫu nhiên", x, x_ref4))

        
        A = [[1, 1], [1, 1]]
        b = [1.0, 2.0]
        x, _ = solve_gauss(A, b)
        g_results.append(check("G5: Hệ vô nghiệm", x, "special"))

    else:
        print("  (Bỏ qua — DinhThuc.py không khả dụng)")
        g_results = [None] * 5

    results["solve_gauss"] = g_results

   
    print("\n" + "=" * 65)
    print("  TEST 2/3 — solve_svd (SVD pseudo-inverse)")
    print("=" * 65)
    s_results = []

    if _SVD_AVAILABLE:
       
        A = [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]]
        b = [8.0, -11.0, -3.0]
        x, info = solve_svd(A, b)
        s_results.append(check("S1: 3×3 nghiệm duy nhất", x, [2.0, 3.0, -1.0]))

       
        A = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        b = [5.0, -3.0, 2.0]
        x, _ = solve_svd(A, b)
        s_results.append(check("S2: Ma trận đơn vị", x, [5.0, -3.0, 2.0]))

        
        A = [[1, 1], [1, 2], [1, 3], [1, 4]]
        b = [6.0, 5.0, 7.0, 10.0]
        x, _ = solve_svd(A, b)
        x_ref_ls = np.linalg.lstsq(np.array(A, float), np.array(b, float), rcond=None)[0].tolist()
        s_results.append(check("S3: Overdetermined least-squares", x, x_ref_ls))

        
        np.random.seed(7)
        A4 = (np.random.rand(4, 4) * 10).tolist()
        A4[0][0] += 15         
        b4 = (np.random.rand(4) * 5).tolist()
        x, _ = solve_svd(A4, b4)
        x_ref4 = np.linalg.solve(np.array(A4, float), np.array(b4, float)).tolist()
        s_results.append(check("S4: 4×4 ngẫu nhiên full-rank", x, x_ref4))

       
        A = [[3, 0, 0], [0, 5, 0], [0, 0, 2]]
        b = [9.0, 10.0, 4.0]
        x, info = solve_svd(A, b)
        s_results.append(check("S5: Ma trận đường chéo", x, [3.0, 2.0, 2.0]))

    else:
        print("  (Bỏ qua — decomposition.py không khả dụng)")
        s_results = [None] * 5

    results["solve_svd"] = s_results

  
    print("\n" + "=" * 65)
    print("  TEST 3/3 — solve_gauss_seidel (phương pháp lặp)")
    print("=" * 65)
    gs_results = []

   
    A = [[10, -1,  2],
         [-1, 11, -1],
         [ 2, -1, 10]]
    b = [6.0, 25.0, -11.0]
    x_ref = np.linalg.solve(np.array(A, float), np.array(b, float)).tolist()
    x, info = solve_gauss_seidel(A, b)
    gs_results.append(check("GS1: 3×3 chéo trội", x, x_ref, info["converged"]))

  
    A = [[20,  1, -1,  2],
         [ 1, 20,  1, -2],
         [-1,  1, 20,  1],
         [ 2, -2,  1, 20]]
    b = [17.0, 20.0, 21.0, 21.0]
    x_ref = np.linalg.solve(np.array(A, float), np.array(b, float)).tolist()
    x, info = solve_gauss_seidel(A, b)
    gs_results.append(check("GS2: 4×4 chéo trội", x, x_ref, info["converged"]))

    
    A = [[5, -1], [-1,  4]]
    b = [7.0, 9.0]
    x_ref = np.linalg.solve(np.array(A, float), np.array(b, float)).tolist()
    x, info = solve_gauss_seidel(A, b, x0=[5.0, 5.0])
    gs_results.append(check("GS3: x0 tùy chọn", x, x_ref, info["converged"]))

    
    print("  --- GS4: Ma trận KHÔNG chéo trội (force=True) ---")
    A = [[1, 3, -2],
         [2, 1,  1],
         [3, 1,  2]]
    b = [5.0, 6.0, 7.0]
    x, info = solve_gauss_seidel(A, b, max_iter=500, force=True)
    
    ok_type = isinstance(x, list) and isinstance(info, dict)
    print(f"    {'✓ PASS' if ok_type else '✗ FAIL'}  [GS4: Không chéo trội]"
          f"  converged={info['converged']}  iter={info['iterations']}")
    gs_results.append(ok_type)

   
    n5 = 5
    np.random.seed(0)
    off5 = np.random.rand(n5, n5) * 0.5
    A5   = off5.tolist()
    for i in range(n5):
        row_sum = sum(abs(off5[i][j]) for j in range(n5) if j != i)
        A5[i][i] = row_sum + 1.5  
    b5   = (np.random.rand(n5) * 10).tolist()
    x_ref5 = np.linalg.solve(np.array(A5, float), np.array(b5, float)).tolist()
    x, info = solve_gauss_seidel(A5, b5)
    gs_results.append(check("GS5: 5×5 chéo trội", x, x_ref5, info["converged"]))

    results["solve_gauss_seidel"] = gs_results

   
    print("\n" + "=" * 65)
    print("  TEST bổ sung — is_strictly_diag_dominant")
    print("=" * 65)

    def check_dom(label, A, expected):
        got, margins = is_strictly_diag_dominant(A)
        ok = (got == expected)
        print(f"    {'✓ PASS' if ok else '✗ FAIL'}  [{label}]"
              f"  expected={expected}  got={got}"
              f"  margins={[round(m, 4) for m in margins]}")
        return ok

    check_dom("DOM1: chéo trội rõ ràng",
              [[10, 1, 1], [1, 10, 1], [1, 1, 10]], True)
    check_dom("DOM2: không chéo trội",
              [[1, 3, -2], [2, 1, 1], [3, 1, 2]],  False)
    check_dom("DOM3: đúng bằng (không chặt)",
              [[2, 1, 1], [1, 2, 1], [1, 1, 2]],   False)  
    check_dom("DOM4: 2×2 chéo trội",
              [[5, 1], [1, 5]], True)
    check_dom("DOM5: 1×1 luôn chéo trội",
              [[3]], True)

    
    print("\n" + "=" * 65)
    print("  TỔNG KẾT")
    print("=" * 65)
    total_pass = total_fail = total_skip = 0
    for solver, res_list in results.items():
        passed = sum(1 for r in res_list if r is True)
        failed = sum(1 for r in res_list if r is False)
        skipped= sum(1 for r in res_list if r is None)
        total_pass += passed
        total_fail += failed
        total_skip += skipped
        print(f"  {solver:30s}  pass={passed}  fail={failed}  skip={skipped}")
    print(f"\n  {'Tổng cộng':30s}  pass={total_pass}  fail={total_fail}  skip={total_skip}")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    _run_tests()
