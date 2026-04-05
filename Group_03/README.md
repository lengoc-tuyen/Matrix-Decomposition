# Matrix Decomposition Project

Đây là đồ án hiện thực các thuật toán nền tảng của đại số tuyến tính bằng Python, tập trung vào khử Gauss, giải hệ phương trình, tính định thức, tìm ma trận nghịch đảo, xác định hạng và cơ sở, diagonalization, và phân rã SVD.

## Mục tiêu

- Hiện thực các thuật toán from scratch, không phụ thuộc thư viện đại số tuyến tính cấp cao.
- Tổ chức mã nguồn theo từng phần rõ ràng để dễ kiểm thử và báo cáo.
- Cung cấp một file chạy tổng để kiểm tra toàn bộ kết quả nhanh chóng.

## Tính năng chính

### Part 1 - Gaussian elimination và các ứng dụng

- `gaussian_eliminate(A, b)`: khử Gauss với chọn pivot để đưa hệ về dạng tam giác trên.
- `back_substitution(U, c)`: giải hệ tam giác trên và tự nhận diện ba trường hợp: nghiệm duy nhất, vô số nghiệm, vô nghiệm.
- `determinant(A)`: tính định thức bằng cách tái sử dụng kết quả khử Gauss.
- `inverse(A)`: tính ma trận nghịch đảo bằng cách giải lần lượt các hệ `Ax = e_i`.
- `rank_and_basis(A)`: xác định rank, cơ sở của không gian cột, hàng và không gian nghiệm.

### Part 2 - Phân rã ma trận

- `diagonalize(A)`: chéo hóa ma trận đối xứng theo dạng `P * D * P_inv`.
- `jacobi_eigen(A)`: tìm trị riêng và vector riêng của ma trận đối xứng bằng lặp QR.
- `svd_decomposition(A)`: phân rã SVD `A = U * Sigma * V^T` dựa trên eigen-decomposition của `A^T A`.
- `utils.py`: cung cấp các hàm dùng chung như `transpose`, `multiply_matrix`, `multiply_mat_vec`, `create_zero_matrix`.

### Part 3 - Kiểm chứng

- `verify_solution(A, x, b)`: kiểm tra sai số của nghiệm `Ax = b`.
- `benchmark.py`: file dành cho đo hiệu năng và thử nghiệm mở rộng.

## Cấu trúc thư mục

```text
Group_03/
├── main.py
├── part1/
│   ├── gaussian.py
│   ├── determinant.py
│   ├── inverse.py
│   ├── rank_basis.py
│   └── part1_demo.ipynb
├── part2/
│   ├── utils.py
│   ├── diagonalization.py
│   ├── decomposition.py
│   └── manim_scene.py
├── part3/
│   ├── solvers.py
│   ├── benchmark.py
│   └── analysis.ipynb
└── report/
	└── report.tex
```

## Yêu cầu môi trường

- Python 3.10 trở lên.
- `numpy` cho phần kiểm chứng nghiệm trong `part3/solvers.py`.

Nếu môi trường của bạn chưa có `numpy`, cài bằng:

```bash
pip install numpy
```

## Cách chạy

Chạy toàn bộ chương trình kiểm thử từ thư mục `Group_03`:

```bash
python3 main.py
```

Nếu cần chỉ định Python cài qua Homebrew:

```bash
/opt/homebrew/bin/python3 main.py
```

## Nội dung kiểm thử của `main.py`

File `main.py` được cấu hình để chạy tuần tự các bài test sau:

1. Kiểm tra `transpose` trong `part2/utils.py`.
2. Kiểm tra khử Gauss và thế ngược.
3. Kiểm tra tính định thức, ma trận nghịch đảo và rank/basis.
4. Kiểm tra diagonalization với ma trận đối xứng.
5. Kiểm tra SVD và khả năng tái tạo lại ma trận gốc.
6. Kiểm tra độ chính xác nghiệm bằng `verify_solution`.

## Ghi chú triển khai

- `main.py` là điểm vào duy nhất để chạy nhanh toàn bộ đồ án.
- Các hàm trong `part1` và `part2` được viết để tái sử dụng lẫn nhau, tránh trùng lặp logic.
- Một số file như `manim_scene.py`, `benchmark.py` và notebook vẫn có thể được mở rộng thêm cho phần trình bày hoặc đo hiệu năng.