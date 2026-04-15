import numpy as np
import time
import json
import os


from solver import solve_gauss, solve_svd, solve_gauss_seidel


def generate_random_system(n):
    """Sinh ma trận ngẫu nhiên (Well-conditioned thông thường)."""
    A = np.random.rand(n, n)
    b = np.random.rand(n)
    return A, b

def generate_hilbert_system(n):
    """
    Sinh ma trận Hilbert H_n (Ill-conditioned).
    Số điều kiện cực kỳ lớn, dễ gây sai số khủng khiếp khi n tăng.
    """
    i, j = np.indices((n, n))
    A = 1.0 / (i + j + 1.0)
    b = np.random.rand(n)
    return A, b

def generate_spd_system(n):
    """
    Sinh ma trận Đối xứng Xác định dương (SPD).
    Luôn thỏa mãn điều kiện chéo trội chặt hàng cho Gauss-Seidel.
    """
    M = np.random.rand(n, n)
    A = np.dot(M, M.T) + n * np.eye(n)
    b = np.random.rand(n)
    return A, b



def calculate_relative_error(A, x_hat, b):
    """Tính sai số tương đối: ||Ax_hat - b||_2 / ||b||_2"""
    A_np = np.array(A, dtype=float)
    x_hat_np = np.array(x_hat, dtype=float)
    b_np = np.array(b, dtype=float)
    
    numerator = np.linalg.norm(np.dot(A_np, x_hat_np) - b_np, 2)
    denominator = np.linalg.norm(b_np, 2)
    
    if denominator == 0:
        return float('inf')
    return numerator / denominator

def run_benchmark():
    """Chạy kịch bản đo lường toàn diện."""
    sizes = [5, 10, 20,50,100]
    
    matrix_generators = {
        'Random': generate_random_system,
        'Hilbert': generate_hilbert_system,
        'SPD': generate_spd_system
    }
    
   
    solvers = {
        'Gauss (Partial Pivot)': {'func': solve_gauss, 'kwargs': {}},
        'SVD (Pseudo-inverse)': {'func': solve_svd, 'kwargs': {}},
        
        'Gauss-Seidel': {'func': solve_gauss_seidel, 'kwargs': {'force': True, 'max_iter': 1000}}
    }
    
    results = []
    
    for n in sizes:
        print(f"\n{'='*50}\n[*] ĐANG ĐO LƯỜNG KÍCH THƯỚC N = {n}\n{'='*50}")
        
        for m_name, generator in matrix_generators.items():
            print(f"\n  [+] Loại ma trận: {m_name}")
            A, b = generator(n)
            
            
            A_list = A.tolist()
            b_list = b.tolist()
            
            for s_name, s_config in solvers.items():
                print(f"      -> Chạy {s_name}...", end="", flush=True)
                
                func = s_config['func']
                kwargs = s_config['kwargs']
                
                total_time = 0.0
                runs = 5
                success_runs = 0
                x_hat = None
                
                for _ in range(runs):
                    start_time = time.perf_counter()
                    try:
                       
                        x_result, info = func(A_list, b_list, **kwargs)
                        
                        
                        if isinstance(x_result, list) and isinstance(x_result[0], float):
                            x_hat = x_result
                            success_runs += 1
                        else:
                            x_hat = None 
                    except Exception as e:
                        
                        x_hat = None
                        break 
                        
                    end_time = time.perf_counter()
                    total_time += (end_time - start_time)
                
                
                if x_hat is not None and success_runs > 0:
                    avg_time = total_time / success_runs
                    error = calculate_relative_error(A, x_hat, b)
                    print(f" Xong! (Thời gian: {avg_time:.4f}s, Sai số: {error:.2e})")
                else:
                    avg_time = None
                    error = None
                    print(" THẤT BẠI HOẶC LỖI SỐ HỌC!")
                    
               
                record = {
                    'size': n,
                    'matrix_type': m_name,
                    'solver': s_name,
                    'avg_time_sec': avg_time,
                    'relative_error': error
                }
                results.append(record)
                
    return results


def main():
    print("=== BẮT ĐẦU BENCHMARK ĐỒ ÁN 1 ===")
    print("Lưu ý: Kích thước n=1000 chạy bằng code Python thuần có thể mất nhiều thời gian.")
    
    data = run_benchmark()
    
    output_file = 'benchmark_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        
    print(f"\n[OK] Đã hoàn thành Benchmark!")
    print(f"[OK] Dữ liệu được lưu tại: {output_file}")
   

if __name__ == '__main__':
    main()