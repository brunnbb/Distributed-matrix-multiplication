from multiprocessing import Pool, cpu_count, shared_memory
import numpy as np
from numba import njit
import test

from utils import read_matrix, save_matrix, benchmark, verify_matrix

@njit()
def block_multiply(A_block, B_block):
    block_size = A_block.shape[0]
    C_block = np.zeros((block_size, block_size))
    for i in range(block_size):
        for j in range(block_size):
            for k in range(block_size):
                C_block[i, j] += A_block[i, k] * B_block[k, j]
    return C_block

def worker(block_indices, A_shm_name, B_shm_name, C_shm_name, block_size, A_shape, B_shape):
    # Acessar memória compartilhada
    A_shm = shared_memory.SharedMemory(name=A_shm_name)
    B_shm = shared_memory.SharedMemory(name=B_shm_name)
    C_shm = shared_memory.SharedMemory(name=C_shm_name)

    A = np.ndarray(A_shape, dtype=np.float64, buffer=A_shm.buf)
    B = np.ndarray(B_shape, dtype=np.float64, buffer=B_shm.buf)
    C = np.ndarray((A_shape[0], B_shape[1]), dtype=np.float64, buffer=C_shm.buf)  # Ajustado para A_shape[0], B_shape[1]

    i, j = block_indices
    C_block = np.zeros((block_size, block_size))

    for k in range((A_shape[1] + block_size - 1) // block_size):  # Ajuste para trabalhar com A.shape[1] (número de colunas de A)
        A_block = A[i * block_size:(i + 1) * block_size, k * block_size:(k + 1) * block_size]
        B_block = B[k * block_size:(k + 1) * block_size, j * block_size:(j + 1) * block_size]
        C_block += block_multiply(A_block, B_block)

    C[i * block_size:(i + 1) * block_size, j * block_size:(j+1) * block_size] = C_block

def block_matrix_multiply(A, B, block_size=64, num_cores=1):
    # Ajustes para matrizes não quadradas
    A_shape = A.shape
    B_shape = B.shape
    assert A_shape[1] == B_shape[0], "Dimensões das matrizes não são compatíveis."
    
    num_blocks_A = (A_shape[0] + block_size - 1) // block_size
    num_blocks_B = (B_shape[1] + block_size - 1) // block_size

    if num_cores is None:
        num_cores = cpu_count()
    num_cores = min(num_cores, cpu_count())

    A_shm = shared_memory.SharedMemory(create=True, size=A.nbytes)
    B_shm = shared_memory.SharedMemory(create=True, size=B.nbytes)
    C_shm = shared_memory.SharedMemory(create=True, size=A_shape[0] * B_shape[1] * np.dtype(np.float64).itemsize)

    # Copiar os dados para a memória compartilhada
    A_shared = np.ndarray(A.shape, dtype=np.float64, buffer=A_shm.buf)
    B_shared = np.ndarray(B.shape, dtype=np.float64, buffer=B_shm.buf)
    np.copyto(A_shared, A)
    np.copyto(B_shared, B)

    # Criar uma lista de índices de blocos a serem processados
    block_indices = [(i, j) for i in range(num_blocks_A) for j in range(num_blocks_B)]

    with Pool(processes=num_cores) as pool:
        pool.starmap(worker, [(idx, A_shm.name, B_shm.name, C_shm.name, block_size, A_shape, B_shape) for idx in block_indices])

    C_result = np.copy(np.ndarray((A_shape[0], B_shape[1]), dtype=np.float64, buffer=C_shm.buf))

    A_shm.close()
    B_shm.close()
    C_shm.close()
    A_shm.unlink()
    B_shm.unlink()
    C_shm.unlink()

    return C_result

def test_multiple_parameters(A_name="matA.txt", B_name="matB.txt", block_sizes=[64, 128, 256, 512, 1024], core_counts=[2, 4, 6, 8, 10, 12]):
    A = read_matrix(A_name)
    B = read_matrix(B_name)
    C = np.zeros((A.shape[0], B.shape[1]))    
    for block_size in block_sizes:
        for num_cores in core_counts:
            print(f"Block Size: {block_size}, Cores: {num_cores}")
            C = benchmark("Multi-threaded", A, B, block_matrix_multiply, block_size=block_size, num_cores=num_cores)    
            verify_matrix(A, B, C)
            
    print("Benchmarking complete.")
    #save_matrix(C, "matrixC.txt")

def main():
    A = np.random.rand(2048, 4096)
    B = np.random.rand(4096, 4096)  # Usando matrizes não quadradas
    C = benchmark("Multi-threaded", A, B, block_matrix_multiply, block_size=128, num_cores=10)
    print("Benchmarking complete.")    
    verify_matrix(A, B, C)
    save_matrix(C, "matC.txt")

if __name__ == "__main__":
    print("Running...")
    main()
    #test_multiple_parameters(block_sizes=[128, 256, 512], core_counts=[8, 10, 12])
    print("Done!")
