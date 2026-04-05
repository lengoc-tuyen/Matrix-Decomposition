import math
from utils import * 

def householder_qr(A: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
    m = len(A)
    n = len(A[0])
    R = [row[:] for row in A]
    Q = create_identity_matrix(m)
    for j in range(n):
        x = [R[i][j] for i in range(j, m)]
        normX = norm(x)
        if R[j][j] >= 0:
            s = -1.0
        else:
            s = 1.0
        u1 = R[j][j] - s * normX
        w = [val / u1 for val in x]
        w[0] = 1.0
        tau = -s * u1 / norm(x)
        for col in range(j, n):
            dot = sum(w[k] * R[j+k][col] for k in range(len(w))) # 
            for k in range(len(w)):
                R[j+k][col] -= tau * dot * w[k] # 

        for row in range(m):
            dot = sum(Q[row][j+k] * w[k] for k in range(len(w))) # 
            for k in range(len(w)):
                Q[row][j+k] -= dot * tau * w[k] # 

    return Q, R

def jacobi_eigen(A: list[list[float]], tol: float = 1e-9) -> tuple[list[float], list[list[float]]]:
    """
    Nhiệm vụ: Tìm Trị riêng và Vector riêng bằng thuật toán QR lặp (thay cho Jacobi).
    
    Các bước logic:
    1. Khởi tạo ma trận Ak là bản sao của A, ma trận V là ma trận đơn vị (để chứa vector riêng).
    2. Vòng lặp lặp lại (ví dụ 100 lần hoặc đến khi hội tụ):
        a. Gọi hàm householder_qr(Ak) để lấy Q và R.
        b. Cập nhật Ak = R * Q (Ma trận này sẽ dần hội tụ về dạng đường chéo).
        c. Cập nhật V = V * Q (Tích các ma trận Q chính là các vector riêng).
        d. Kiểm tra nếu các phần tử ngoài đường chéo của Ak nhỏ hơn tol thì break.
    3. Trị riêng là các phần tử trên đường chéo của Ak.
    4. Trả về (danh sách eigenvalues, ma trận eigenvectors V).
    """ 
    Ak = [row[:] for row in A]
    V = create_identity_matrix(len(A))
    for i in range(100):
        Q, R = householder_qr(Ak)
        Ak = multiply_matrix(R, Q)
        V = multiply_matrix(V, Q)
        off_diag_norm = 0
        n = len(Ak)
        for r in range(n):
            for c in range(n):
                if r != c:
                    off_diag_norm += Ak[r][c] ** 2
        if math.sqrt(off_diag_norm) < tol:
            break
    eigenvalues = [Ak[i][i] for i in range(len(Ak))]
    return eigenvalues, V
    



def diagonalize(A: list[list[float]]) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    """
    Nhiệm vụ: Phân tích A thành P * D * P_transpose (Yêu cầu 2.1).

    1. Gọi jacobi_eigen(A) để lấy eigenvalues và ma trận P.
    2. Tạo ma trận đường chéo D từ danh sách eigenvalues.
    3. Tính P_transpose (vì P là ma trận trực giao nên nghịch đảo là chuyển vị).
    4. Trả về (P, D, P_transpose).
    """
    eigenvalues, P = jacobi_eigen(A)
    l = len(eigenvalues)
    D = [[eigenvalues[i] if i == j else 0.0 for j in range(l)] for i in range(l)]
    P_transpose = transpose(P)
    return (P, D, P_transpose)
