from determinant import determinant
from gaussian import gaussian_eliminate


def inverse(A):
    """
    Tính A^{-1} bằng cách tái sử dụng gaussian_eliminate.
    Giải A xᵢ = eᵢ cho từng cột đơn vị eᵢ → ghép lại thành A^{-1}.
    """
    n = len(A)
    for row in A:
        if len(row) != n:
            raise ValueError("Ma trận phải là ma trận vuông.")

    # Kiểm tra khả nghịch trước qua định thức
    det_val = determinant(A)
    if abs(det_val) < 1e-10:
        return "Ma trận KHÔNG KHẢ NGHỊCH (det(A) = 0)."


    A_inv_cols = []


    for i in range(n):
        # Cột đơn vị thứ i: e_i = [0,...,1,...,0]
        e_i = [1.0 if j == i else 0.0 for j in range(n)]


        # Tái sử dụng gaussian_eliminate để giải A x_i = e_i
        _, x_i, _ = gaussian_eliminate(A, e_i)


        A_inv_cols.append(x_i)


    # Chuyển từ danh sách n cột → danh sách n dòng
    A_inv = [[A_inv_cols[j][i] for j in range(n)] for i in range(n)]
    return A_inv
