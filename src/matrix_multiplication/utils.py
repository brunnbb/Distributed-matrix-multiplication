import time
import logging
import random
import hashlib
from typing import Callable

import psutil
import numpy as np
from numpy.typing import NDArray

MATRIX_PATH = "src\\matrix_multiplication\\data\\"
LOG_PATH = "src\\matrix_multiplication\\logs\\"

def read_matrix(name: str) -> NDArray:
    return np.loadtxt(f"{MATRIX_PATH}{name}", dtype=np.float64)

def save_matrix(M: NDArray, name: str) -> None:
    np.savetxt(f"{MATRIX_PATH}{name}", M, fmt="%.4f")
    
def benchmark(mult_type: str, A: NDArray, B: NDArray, mult: Callable, **kwargs) -> NDArray:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(message)s",
        filename=f"{LOG_PATH}time_perf.log"
    )
    
    clock_start = time.perf_counter()
    cpu_start = time.process_time()
    C = mult(A, B, **kwargs)
    cpu_end = time.process_time()
    clock_end = time.perf_counter()
    
    log_message = (
        f" {mult_type:<15} " 
        f"{str(A.shape):<12} X {str(B.shape):<12} "  
        f"| Clock time = {clock_end - clock_start:8.4f} seconds "  
        f"| CPU time = {cpu_end - cpu_start:8.4f} seconds "  
        f"| {str(kwargs):<30}" 
    )
    
    logging.info(log_message)    
    return C

def verify_matrix(A: NDArray, B: NDArray, C: NDArray) -> None:
    C1 = np.dot(A, B)
    print("\n--- Verification ---")
    if np.allclose(C, C1, rtol=1e-12, atol=1e-12):
        print("Close match between computed and numpy.dot results")
    else:
        print(f"Mismatch detected between computed and numpy.dot results")
    max_diff = np.max(np.abs(C - C1))
    print(f"Maximum error between computed and numpy.dot: {max_diff:.16f}")

def generate_matrix() -> None:
    mat_size = int(input("\tType the matrix size: "))
    fA = open(f"{MATRIX_PATH}matrixA.txt", "w", encoding="utf8")
    fB = open(f"{MATRIX_PATH}matrixB.txt", "w", encoding="utf8")
    for lin in range(mat_size):
        for col in range(mat_size):
            fA.writelines(str(round(random.uniform(0.15, 1.15), 2)))
            fB.writelines(str(round(random.uniform(0.15, 1.15), 2)))
            if (col + 1 < mat_size):
                fA.writelines(" ")
                fB.writelines(" ")
        if (lin + 1 < mat_size):
            fA.writelines("\n")
            fB.writelines("\n")
    fA.close()
    fB.close()

def psutil_test() -> None:
    print("CPU use:", psutil.cpu_percent(), "%")
    print("Logical cores:", psutil.cpu_count())
    print("Cores:", psutil.cpu_count(logical=False))

def verify_hash(file_name: str, expected_hash: str) -> bool:
    with open(f"{MATRIX_PATH}{file_name}", "rb") as f:
        file_bytes = f.read()
        mat_hash = hashlib.sha256(file_bytes).hexdigest()
    print(f"SHA-256 hash of {file_name}: {mat_hash}")
    return mat_hash == expected_hash

if __name__ == "__main__":
    print(verify_hash("matC.txt", "b1c9746749e75c15c1c6e398bb77618db177333b3e1e7b6c72540800ebbe956f"))
    #psutil_test()
    #generate_matrix()