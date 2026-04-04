import numpy as np


def verify_solution(A, x, b):
	# Chuyển về mảng NumPy kiểu float để tính toán chính xác
	A = np.array(A, dtype=float)
	x = np.array(x, dtype=float)
	b = np.array(b, dtype=float)

	# Tính Ax (tích ma trận - vector)
	Ax = A @ x

	# Tính sai số tuyệt đối: ||Ax - b||
	numerator = np.linalg.norm(Ax - b)

	# Tính chuẩn của b: ||b||
	denominator = np.linalg.norm(b)

	# Tránh chia cho 0 (trường hợp b ≈ 0)
	if denominator < 1e-12:
		return numerator  # trả về sai số tuyệt đối

	# Sai số tương đối: ||Ax - b|| / ||b||
	return numerator / denominator
