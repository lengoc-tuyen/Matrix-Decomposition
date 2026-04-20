def inverse(A):
    """
    Tính ma trận nghịch đảo A^{-1} bằng phương pháp Gauss-Jordan
    biến đổi đồng thời trên ma trận ghép [A | I_n].
    """
    n = len(A)
    # Kiểm tra ma trận vuông
    for row in A:
        if len(row) != n:
            raise ValueError("Ma trận phải là ma trận vuông.")

    # 1. Tạo ma trận ghép [A | I_n]
    Ag = []
    for i in range(n):
        row = list(A[i]) # Copy dòng của A
        # Thêm ma trận đơn vị
        for j in range(n):
            if i == j:
                row.append(1.0)
            else:
                row.append(0.0)
        Ag.append(row)

    # 2. Khử Gauss để đưa phần bên trái về dạng tam giác trên
    for i in range(n):
        # Chọn pivot cục bộ (partial pivoting)
        max_idx = i
        for k in range(i + 1, n):
            if abs(Ag[k][i]) > abs(Ag[max_idx][i]):
                max_idx = k
        
        # Nếu pivot = 0 -> ma trận suy biến
        if abs(Ag[max_idx][i]) < 1e-10:
            return "Ma trận KHÔNG KHẢ NGHỊCH (det(A) = 0)."
            
        # Hoán vị dòng
        if max_idx != i:
            Ag[i], Ag[max_idx] = Ag[max_idx], Ag[i]
            
        # Chia dòng i cho pivot
        pivot = Ag[i][i]
        for j in range(i, 2*n):
            Ag[i][j] /= pivot
            
        # Khử các dòng dưới
        for k in range(i + 1, n):
            factor = Ag[k][i]
            for j in range(i, 2*n):
                Ag[k][j] -= factor * Ag[i][j]

    # 3. Khử Gauss-Jordan để đưa phần bên trái về ma trận đơn vị I_n (Thế ngược)
    for i in range(n - 1, -1, -1):
        for k in range(i - 1, -1, -1):
            factor = Ag[k][i]
            for j in range(i, 2*n):
                Ag[k][j] -= factor * Ag[i][j]

    # 4. Trích xuất ma trận A^{-1} từ nửa phải của [I_n | A^{-1}]
    A_inv = []
    for i in range(n):
        A_inv.append(Ag[i][n:])
        
    return A_inv
