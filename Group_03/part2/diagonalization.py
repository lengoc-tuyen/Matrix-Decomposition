try:
    from .utils import create_zero_matrix, transpose
except ImportError:
    from utils import create_zero_matrix, transpose


def jacobi_eigen(A: list[list[float]], tol: float = 1e-9) -> tuple[list[float], list[list[float]]]:
    """
    Nhiệm vụ: Tìm tất cả Trị riêng và Vector riêng của ma trận đối xứng A.
    - Input: Ma trận đối xứng A.
    - Output: (danh sách eigenvalues, ma trận eigenvectors P).
    Gợi ý: Dùng vòng lặp tìm phần tử lớn nhất ngoài đường chéo và thực hiện quay Jacobi.
    """
    pass

def diagonalize(A: list[list[float]]) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    """
    Nhiệm vụ: Phân tích A thành P * D * P_inv (Yêu cầu 2.1 của đồ án).
    - Output: (Ma trận P, Ma trận đường chéo D, Ma trận P nghịch đảo).
    - Lưu ý: Vì dùng Jacobi cho ma trận đối xứng, P_inv chính là transpose của P.
    """
    pass