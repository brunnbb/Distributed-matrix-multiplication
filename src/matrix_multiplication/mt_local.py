from multiprocessing import Pool, cpu_count, shared_memory

import numpy as np
from numba import njit

from utils import read_matrix, save_matrix, benchmark, verify_matrix, save_stats_json

@njit()
def multiply_block(A_block: np.ndarray, B_block: np.ndarray) -> np.ndarray:
    C_block = np.zeros((A_block.shape[0], B_block.shape[1]), dtype=np.float64)
    for i in range(A_block.shape[0]):
        for j in range(B_block.shape[1]):
            for k in range(A_block.shape[1]):
                C_block[i,j] += A_block[i, k] * B_block[k, j]
    return C_block

def worker(block_indices: tuple[int, int], A_shm_name: str, B_shm_name: str, C_shm_name: str, block_size: int, A_shape: tuple[int, int], B_shape: tuple[int, int]) -> None:
    # Access shared memory
    A_shm = shared_memory.SharedMemory(name=A_shm_name)
    B_shm = shared_memory.SharedMemory(name=B_shm_name)
    C_shm = shared_memory.SharedMemory(name=C_shm_name)

    A = np.ndarray(A_shape, dtype=np.float64, buffer=A_shm.buf)
    B = np.ndarray(B_shape, dtype=np.float64, buffer=B_shm.buf)
    C = np.ndarray((A_shape[0], B_shape[1]), dtype=np.float64, buffer=C_shm.buf) 

    i, j = block_indices
    num_blocks_k = (A_shape[1] + block_size - 1) // block_size
    
    # Compute row/col ranges for this block
    row_start = i * block_size
    row_end = min((i + 1) * block_size, A_shape[0])
    col_start = j * block_size
    col_end = min((j + 1) * block_size, B_shape[1])
    
    # Shape of result block
    out_row = row_end - row_start
    out_col = col_end - col_start
    C_block = np.zeros((out_row, out_col), dtype=np.float64)

    for k in range(num_blocks_k):
        a_col_start = k * block_size
        a_col_end = min((k + 1) * block_size, A_shape[1])
        
        # Slices for this iteration
        A_block = A[row_start:row_end, a_col_start:a_col_end]
        B_block = B[a_col_start:a_col_end, col_start:col_end]
        if A_block.size == 0 or B_block.size == 0:
            continue
        
        C_block += multiply_block(A_block, B_block)

    C[row_start:row_end, col_start:col_end] = C_block
    
    A_shm.close(); B_shm.close(); C_shm.close()

def block_matrix_multiply(A: np.ndarray, B: np.ndarray, block_size=64, num_cores=1) -> np.ndarray:
    A_shape: tuple[int, int] = A.shape
    B_shape: tuple[int, int] = B.shape
    assert A_shape[1] == B_shape[0], "Incompatible matrix dimensions for multiplication."
    
    num_blocks_A: int = (A_shape[0] + block_size - 1) // block_size
    num_blocks_B: int = (B_shape[1] + block_size - 1) // block_size

    if num_cores is None:
        num_cores = cpu_count()
    num_cores = min(num_cores, cpu_count())

    A_shm = shared_memory.SharedMemory(create=True, size=A.nbytes)
    B_shm = shared_memory.SharedMemory(create=True, size=B.nbytes)
    C_shm = shared_memory.SharedMemory(create=True, size=A_shape[0] * B_shape[1] * np.dtype(np.float64).itemsize)

    # Copy data to shared memory
    A_shared = np.ndarray(A.shape, dtype=np.float64, buffer=A_shm.buf)
    B_shared = np.ndarray(B.shape, dtype=np.float64, buffer=B_shm.buf)
    np.copyto(A_shared, A)
    np.copyto(B_shared, B)
    
    # Create list of block indices
    block_indices: list[tuple[int, int]] = [(i, j) for i in range(num_blocks_A) for j in range(num_blocks_B)]

    with Pool(processes=num_cores) as pool:
        pool.starmap(worker, [(idx, A_shm.name, B_shm.name, C_shm.name, block_size, A_shape, B_shape) for idx in block_indices])

    C_result = np.copy(np.ndarray((A_shape[0], B_shape[1]), dtype=np.float64, buffer=C_shm.buf))

    # Clean up shared memory
    A_shm.close(); B_shm.close(); C_shm.close() 
    A_shm.unlink(); B_shm.unlink(); C_shm.unlink()
    
    return C_result

def test_multiple_parameters(A_name: str, B_name: str, block_sizes=[128, 256, 512, 1024], core_counts=[2, 4, 6, 8, 10, 12]):
    A: np.ndarray = read_matrix(A_name)
    B: np.ndarray = read_matrix(B_name)
    C: np.ndarray = np.zeros((A.shape[0], B.shape[1]), dtype=np.float64)    
    
    index: int = 0
    total: int = len(block_sizes) * len(core_counts)
    for block_size in block_sizes:
        for num_cores in core_counts:
            index += 1
            print(f"\n[{index:02}|{total:02}]: Block Size: {block_size}, Cores: {num_cores}")
            C, elapsed_time = benchmark("Blocked local", A, B, block_matrix_multiply, block_size=block_size, num_cores=num_cores)    
            if verify_matrix(A, B, C):
                info = {
                    "Blocked local":{
                        "time_seconds": elapsed_time,
                        "num_cores": num_cores,
                        "block_size": block_size
                    }
                }
                save_stats_json(info)
            save_matrix(C, f"matC_mt{index}.txt")         
    print("\nBenchmark of multiple parameters completed.")

if __name__ == "__main__":
    print("Running...")
    test_multiple_parameters(A_name="matA.txt", B_name="matB.txt", block_sizes=[128, 256, 512], core_counts=[1, 2])
    print("Done!")
