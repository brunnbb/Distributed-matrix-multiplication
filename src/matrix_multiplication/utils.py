import time
import logging
import random
import hashlib
from typing import Callable

import numpy as np

MATRIX_PATH = "src\\matrix_multiplication\\data\\"
LOG_PATH = "src\\matrix_multiplication\\logs\\"

def deflotify_matrix(M: np.ndarray, scale: int) -> np.ndarray:
    M = M * scale
    return M

def flotify_matrix(M: np.ndarray, scale: int) -> np.ndarray:
    M = M / scale ** 2   
    return M

def read_matrix(name: str) -> np.ndarray:
    return np.loadtxt(f"{MATRIX_PATH}{name}", dtype=np.float64)
    
def save_matrix(M: np.ndarray, name: str) -> None:
    np.savetxt(f"{MATRIX_PATH}{name}", M, fmt="%1.4f")

def truncate_matrix(M: np.ndarray, decimals: int = 4) -> np.ndarray:
    factor = 10.0 ** decimals
    return np.trunc(M * factor) / factor
    
def benchmark(mult_type: str, A: np.ndarray, B: np.ndarray, mult: Callable, **kwargs) -> np.ndarray:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(message)s",
        filename=f"{LOG_PATH}time_perf.log"
    )
    
    clock_start = time.perf_counter()
    C = mult(A, B, **kwargs)
    clock_end = time.perf_counter()
    
    log_message = (
        f" {mult_type:<15} " 
        f"{str(A.shape):<12} X {str(B.shape):<12} "  
        f"| Clock time = {clock_end - clock_start:8.4f} seconds "
        f"| {str(kwargs):<30}" 
    )
    
    logging.info(log_message)    
    return C

def verify_matrix(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> None:
    C1 = np.dot(A, B)
    print("\n--- Verification ---")
    print(f"Shape of computed result: {C.shape}, Shape of numpy.dot result: {C1.shape}")
    if np.allclose(C, C1, rtol=1e-12, atol=1e-12):
        print("✅ Close match between computed and numpy.dot results")
    else:
        print(f"❌ Mismatch detected between computed and numpy.dot results")
    max_diff = np.max(np.abs(C - C1))
    print(f"Maximum error between computed and numpy.dot in memory: {max_diff:.14f}\n")

def generate_matrix() -> None:
    mat_size = int(input("\tType the matrix size: "))
    fA = open(f"{MATRIX_PATH}mtest_a.txt", "w", encoding="utf8")
    fB = open(f"{MATRIX_PATH}mtest_b.txt", "w", encoding="utf8")
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

def verify_hash(file_name: str, expected_hash: str = "") -> bool:
    with open(f"{MATRIX_PATH}{file_name}", "rb") as f:
        file_bytes = f.read()
        mat_hash = hashlib.sha256(file_bytes).hexdigest()
    print(f"SHA-256 hash of {file_name}: {mat_hash}")
    return mat_hash == expected_hash

def show_matrix_differences(A: np.ndarray, B: np.ndarray, tol: float = 0.0) -> None:
    if A.shape != B.shape:
        print(f"⚠️ Matrices have different shapes: {A.shape} vs {B.shape}")
        return

    diff_mask = np.abs(A - B) > tol

    if not np.any(diff_mask):
        print("✅ Matrices are identical (within tolerance).")
        return

    print(f"🔍 Found {np.sum(diff_mask)} differing elements:\n")
    rows, cols = np.where(diff_mask)

    for r, c in zip(rows, cols):
        print(f"→ Position ({r}, {c}): A={A[r, c]} | B={B[r, c]}")

# Only exists for temporary testing purposes
def test():
    C1 = read_matrix("matCnp_bruno.txt")
    print("Shape: ", C1.shape)
    
    C2 = read_matrix("matC_mt1.txt")
    print("Shape: ", C2.shape)
    
    print("Max difference between matC_mt1 and matC_mt2:", np.max(np.abs(C1 - C2)))
    
    print(verify_hash("matCnp_bruno.txt", "b1c9746749e75c15c1c6e398bb77618db177333b3e1e7b6c72540800ebbe956f"))
    print(verify_hash("matC_mt1.txt", "b1c9746749e75c15c1c6e398bb77618db177333b3e1e7b6c72540800ebbe956f"))

if __name__ == "__main__":
    test()