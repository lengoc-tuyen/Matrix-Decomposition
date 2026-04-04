def create_zero_matrix(m: int, n: int) -> list[list[float]]:
    A = []
    for i in range(m):
        row = []
        for j in range(n):
            row.append(0)
        A.append(row)
    return A

def transpose(A: list[list[float]]) -> list[list[float]]:
    if not A:
        return []
    
    m = len(A)
    n = len(A[0])
    T = create_zero_matrix(n, m)
    
    for i in range(m):
        for j in range(n):
            T[j][i] = A[i][j]
            
    return T

def multiply_matrix(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    if not A or not B:
        return []
        
    m = len(A)
    p = len(A[0])
    p_B = len(B)
    n = len(B[0])
    
    if p != p_B:
        raise ValueError("Lỗi: Số cột của A phải bằng số hàng của B!")
        
    C = create_zero_matrix(m, n)
    for i in range(m):
        for j in range(n):
            for k in range(p):
                C[i][j] += A[i][k] * B[k][j]
                
    return C

def multiply_mat_vec(A: list[list[float]], v: list[float]) -> list[float]:
    if not A or not v:
        return []
        
    m = len(A)
    n = len(A[0])
    
    if n != len(v):
        raise ValueError("Lỗi: Số cột của ma trận A phải bằng kích thước của vector v!")
        
    result = [0.0] * m 
    for i in range(m):
        for j in range(n):
            result[i] += A[i][j] * v[j]
            
    return result

def get_column(A: list[list[float]], col_index: int) -> list[float]:
    return [row[col_index] for row in A]