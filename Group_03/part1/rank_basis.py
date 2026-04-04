def rank_and_basis(A):
	M = [row[:] for row in A]
	m, n = len(M), len(M[0])
	eps = 1e-10

	pivot_cols = []
	row = 0

	# ===== Gauss-Jordan + Partial Pivoting =====
	for col in range(n):
		# chọn pivot lớn nhất
		pivot = max(range(row, m), key=lambda r: abs(M[r][col]))
		if abs(M[pivot][col]) < eps:
			continue

		# đổi dòng
		M[row], M[pivot] = M[pivot], M[row]

		# chuẩn hóa pivot = 1
		pv = M[row][col]
		M[row] = [x / pv for x in M[row]]

		# khử toàn bộ cột (RREF)
		for r in range(m):
			if r != row and abs(M[r][col]) > eps:
				factor = M[r][col]
				M[r] = [M[r][c] - factor * M[row][c] for c in range(n)]

		pivot_cols.append(col)
		row += 1
		if row == m:
			break

	rank = len(pivot_cols)

	# ===== Column space =====
	col_basis = [[A[i][c] for i in range(m)] for c in pivot_cols]

	# ===== Row space =====
	row_basis = [
		[0 if abs(x) < eps else x for x in r]
		for r in M if any(abs(x) > eps for x in r)
	]

	# ===== Null space =====
	free_cols = [j for j in range(n) if j not in pivot_cols]
	null_basis = []

	for free in free_cols:
		vec = [0.0] * n
		vec[free] = 1.0
		for i, pc in enumerate(pivot_cols):
			val = -M[i][free]
			vec[pc] = 0 if abs(val) < eps else val
		null_basis.append(vec)

	return rank, {
		"column_space": col_basis,
		"row_space": row_basis,
		"null_space": null_basis
	}
