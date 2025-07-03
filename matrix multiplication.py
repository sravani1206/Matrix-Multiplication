import numpy as np
from multiprocessing import Pool, cpu_count
import time
import matplotlib.pyplot as plt

# Function for serial matrix multiplication
def serial_matrix_multiplication(A, B):
    n, m = A.shape
    _, p = B.shape
    result = np.zeros((n, p))
    for i in range(n):
        for j in range(p):
            result[i, j] = sum(A[i, k] * B[k, j] for k in range(m))
    return result

# Function for parallel matrix multiplication
def parallel_worker(args):
    A_row, B = args
    return np.dot(A_row, B)

def parallel_matrix_multiplication(A, B, num_processors):
    n = A.shape[0]
    with Pool(processes=num_processors) as pool:
        result = pool.map(parallel_worker, [(A[i], B) for i in range(n)])
    return np.array(result)

# Function to test performance
def benchmark(matrix_size, processors):
    # Generate random matrices
    A = np.random.rand(matrix_size, matrix_size)
    B = np.random.rand(matrix_size, matrix_size)

    # Serial execution
    start_time = time.time()
    serial_result = serial_matrix_multiplication(A, B)
    serial_time = time.time() - start_time

    # Parallel execution
    parallel_times = []
    for num_proc in processors:
        start_time = time.time()
        parallel_result = parallel_matrix_multiplication(A, B, num_proc)
        parallel_time = time.time() - start_time
        parallel_times.append(parallel_time)

        # Validate that the results are the same
        assert np.allclose(serial_result, parallel_result), "Mismatch in results!"

    return serial_time, parallel_times

# Main function
if __name__ == "__main__":
    matrix_size = 100  
    processors = [1, 2, 4, 8, 16]

    serial_time, parallel_times = benchmark(matrix_size, processors)

    # Calculate speedup and efficiency
    speedup = [serial_time / t for t in parallel_times]
    efficiency = [s / p for s, p in zip(speedup, processors)]

    # Print summary results
    print(f"Matrix Size: {matrix_size}x{matrix_size}")
    print(f"Serial Execution Time: {serial_time:.4f} seconds")
    print(f"Parallel Execution Time (Min): {min(parallel_times):.4f} seconds")
    print(f"Speedup (Best Case): {serial_time / min(parallel_times):.4f}")
    print(f"Efficiency (Best Case): {(serial_time / min(parallel_times)) / max(processors):.4f}")

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(processors, parallel_times, marker='o', label='Parallel Time')
    plt.axhline(y=serial_time, color='r', linestyle='--', label='Serial Time')
    plt.xlabel('Number of Processors')
    plt.ylabel('Time (s)')
    plt.title('Execution Time vs Processors')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(processors, speedup, marker='o', label='Speedup')
    plt.plot(processors, efficiency, marker='o', label='Efficiency')
    plt.xlabel('Number of Processors')
    plt.ylabel('Speedup/Efficiency')
    plt.title('Speedup and Efficiency')
    plt.legend()

    plt.tight_layout()
    plt.show()
