def back_substitution(U, c):
    """
    Giải hệ phương trình tam giác trên Ux = c.
    Tự động nhận diện: Có nghiệm duy nhất, Vô số nghiệm, Vô nghiệm.
    """
    m = len(U)       # Số phương trình
    n = len(U[0])    # Số ẩn
    epsilon = 1e-12
   
    # Bước 1: Quét U để tìm vị trí các phần tử chốt (pivot)
    pivot_pos = {}   # Lưu map: Dòng i -> Cột chứa pivot
    pivot_cols = []  # Danh sách các cột có pivot
   
    for i in range(m):
        pivot_col = -1
        # Tìm phần tử khác 0 đầu tiên trong dòng i (từ trái sang phải)
        for j in range(n):
            if abs(U[i][j]) > epsilon:
                pivot_col = j
                break
               
        if pivot_col != -1:
            pivot_pos[i] = pivot_col
            pivot_cols.append(pivot_col)
        else:
            # Dòng i toàn số 0. Kiểm tra vế phải c[i]
            if abs(c[i]) > epsilon:
                return "Hệ phương trình VÔ NGHIỆM"
               
    # Bước 2: Xử lý dựa trên số lượng pivot
    if len(pivot_cols) == n:
        # Trường hợp 1: NGHIỆM DUY NHẤT (Số pivot = số ẩn)
        x = [0.0] * n
        for i in range(m - 1, -1, -1):
            if i not in pivot_pos:
                continue # Bỏ qua các dòng 0 = 0 ở đáy (nếu có)
               
            p_col = pivot_pos[i]
            sum_ax = 0.0
            for j in range(p_col + 1, n):
                sum_ax += U[i][j] * x[j]
            x[p_col] = (c[i] - sum_ax) / U[i][p_col]
        return x
       
    else:
        # Trường hợp 2: VÔ SỐ NGHIỆM (Số pivot < số ẩn)
        exprs = []
        for j in range(n):
            if j not in pivot_cols:
                exprs.append({'const': 0.0, j: 1.0}) # Ẩn tự do
            else:
                exprs.append({'const': 0.0}) # Ẩn cơ sở
               
        # Thế ngược từ dưới lên để lập công thức
        for i in range(m - 1, -1, -1):
            if i not in pivot_pos:
                continue
               
            p_col = pivot_pos[i]
            pivot_val = U[i][p_col]
            current_expr = {'const': c[i] / pivot_val}
           
            for j in range(p_col + 1, n):
                coef = U[i][j] / pivot_val
                if abs(coef) > epsilon:
                    for key, val in exprs[j].items():
                        current_expr[key] = current_expr.get(key, 0.0) - coef * val
            exprs[p_col] = current_expr
           
        # Format kết quả thành chuỗi dễ đọc
        result = []
        for j in range(n):
            if j not in pivot_cols:
                result.append(f"x_{j} là ẩn tự do (thuộc R)")
            else:
                terms = []
                c_val = exprs[j].get('const', 0.0)
                if abs(c_val) > epsilon or len(exprs[j]) == 1:
                    terms.append(f"{c_val:.3f}")
                   
                for k in range(n):
                    if k in exprs[j] and abs(exprs[j][k]) > epsilon and k != 'const':
                        val = exprs[j][k]
                        if val > 0 and terms:
                            terms.append(f"+ {val:.3f}*x_{k}")
                        elif val > 0:
                            terms.append(f"{val:.3f}*x_{k}")
                        else:
                            terms.append(f"- {abs(val):.3f}*x_{k}")
                           
                if not terms: terms.append("0.000")
                result.append(f"x_{j} = " + " ".join(terms))
        return result


def gaussian_eliminate(A, b):
    m = len(A)
    n = len(A[0])
   
    # 1. Tạo ma trận tăng cường M
    M = []
    for i in range(m):
        row = [float(val) for val in A[i]]
        row.append(float(b[i]))
        M.append(row)
       
    swaps = 0
    epsilon = 1e-12
    r = 0 # Chỉ số dòng
   
    # 2. Quá trình khử Gauss
    for c_col in range(n):
        if r >= m: break
           
        # Tìm pivot
        max_val = 0.0
        p = r
        for i in range(r, m):
            if abs(M[i][c_col]) > max_val:
                max_val = abs(M[i][c_col])
                p = i
               
        # Nếu cột này toàn 0 từ dòng r trở xuống, bỏ qua, xét cột tiếp theo
        if max_val < epsilon:
            print(f"Không có pivot tại cột {c_col}.")
            continue
           
        # Hoán đổi dòng
        if p != r:
            M[r], M[p] = M[p], M[r]
            swaps += 1
           
        # Khử các dòng bên dưới
        for i in range(r + 1, m):
            l_ik = M[i][c_col] / M[r][c_col]
            for j in range(c_col, n + 1):
                M[i][j] -= l_ik * M[r][j]
               
        r += 1
       
    # 3. Trích xuất U và c từ ma trận M đã khử
    U = [row[:n] for row in M]
    c_vec = [row[n] for row in M]
   
    # 4. Gọi hàm thế ngược duy nhất
    x = back_substitution(U, c_vec)
   
    return U, x, swaps
