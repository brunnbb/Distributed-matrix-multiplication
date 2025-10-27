import json
import time
import logging
import random
import hashlib
from typing import Callable

import numpy as np

DATA_PATH = "src\\matrix_multiplication\\data\\"
LOG_PATH = "src\\matrix_multiplication\\logs\\"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
    filename=f"{LOG_PATH}time_perf.log"
)

def read_matrix(name: str, should_log: bool = False) -> np.ndarray:
    try:
        start = time.perf_counter()
        matrix = np.loadtxt(f"{DATA_PATH}{name}", dtype=np.float64)
        end = time.perf_counter()
        if should_log:
            logging.info(f" Time to read matrix {name} from txt: {end - start:.4f} seconds")
        return matrix
    except FileNotFoundError as e:
        logging.error(f" File {name} not found at {DATA_PATH}: {e}")
        raise
    except Exception as e:
        logging.error(f" Error reading matrix {name}: {e}")
        raise
    
def save_matrix(M: np.ndarray, name: str, should_log: bool = False) -> None:
    try:
        start = time.perf_counter()
        np.savetxt(f"{DATA_PATH}{name}", M, fmt="%.8f")
        end = time.perf_counter()
        if should_log:
            logging.info(f" Time to save matrix {name} as a txt: {end - start:.4f} seconds")
    except Exception as e:
        logging.error(f" Failed to save matrix {name} to {DATA_PATH}: {e}")
        raise

def truncate_matrix(M: np.ndarray, decimals: int = 4) -> np.ndarray:
    try:
        factor = 10.0 ** decimals
        return np.trunc(M * factor) / factor
    except Exception as e:
        logging.error(f" Failed to truncate matrix to {decimals} decimals: {e}")
        raise
    
def benchmark(mult_type: str, A: np.ndarray, B: np.ndarray, mult: Callable, **kwargs) -> tuple[np.ndarray, float]:    
    try:
        clock_start = time.perf_counter()
        C = mult(A, B, **kwargs)
        clock_end = time.perf_counter()
        elapsed_time = clock_end - clock_start
        
        log_message = (
            f" {mult_type:<15} " 
            f"{str(A.shape):<12} X {str(B.shape):<12} "  
            f"| Time = {elapsed_time:8.4f} seconds "
            f"| Params: {str(kwargs):<30}" 
        )
        
        logging.info(log_message)    
        return C, elapsed_time
    except Exception as e:
        logging.error(f" Failed to benchmark matrix multiplication type [{mult_type}, {kwargs}]: {e}")
        raise

def save_stats_json(data: dict) -> None:
    try:
        stats = []
        with open(f"{DATA_PATH}stats.json", "r", encoding="utf8") as f:
            try:
                stats = json.load(f)
            except json.JSONDecodeError:
                stats = []
        stats.append(data)
        with open(f"{DATA_PATH}stats.json", "w", encoding="utf8") as f:
            json.dump(stats, f, indent=4)
        logging.info(f" Stats saved to {DATA_PATH}stats.json")
    except Exception as e:
        logging.error(f" Failed to save stats to JSON: {e}")
        raise

def verify_matrix(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> None:
    try:
        C1 = np.dot(A, B)
        print("--- Verification ---")
        print(f"Shape of computed result: {C.shape}, Shape of numpy.dot result: {C1.shape}")
        max_diff = np.max(np.abs(C - C1))
        print(f"Maximum error between computed and numpy.dot in memory: {max_diff:.14f}")
        if np.allclose(C, C1, rtol=1e-12, atol=1e-12):
            print("✅ Close match between computed and numpy.dot results")
            return True
        else:
            print(f"❌ Mismatch detected between computed and numpy.dot results")
            return False
    except Exception as e:
        logging.error(f" Failed to verify matrices: {e}")
        raise 

def generate_matrix() -> None:
    try:
        mat_size = int(input("\tType the matrix size: "))
        with open(f"{DATA_PATH}mtest_a.txt", "w", encoding="utf8") as fA, \
             open(f"{DATA_PATH}mtest_b.txt", "w", encoding="utf8") as fB:
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
    except Exception as e:
        logging.error(f" Failed to generate matrices: {e}")
        raise

def verify_hash(file_name: str, expected_hash: str = "") -> bool:
    try:
        with open(f"{DATA_PATH}{file_name}", "rb") as f:
            file_bytes = f.read()
            mat_hash = hashlib.sha256(file_bytes).hexdigest()
        print(f"SHA-256 hash of {file_name}: {mat_hash}")
        return mat_hash == expected_hash
    except FileNotFoundError as e:
        logging.error(f" File {file_name} not found at {DATA_PATH} during hash verification: {e}")
        raise
    except Exception as e:
        logging.error(f" Failed to compute hash for {file_name}: {e}")
        raise

def show_matrix_differences(A: np.ndarray, B: np.ndarray, tol: float = 0.0) -> None:
    try:
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
    except Exception as e:
        logging.error(f" Failure when comparing matrices: {e}")
        raise

def numpy_test():
    try:
        A = read_matrix("matA.txt")
        B = read_matrix("matB.txt")
        print("Performing Numpy matrix multiplication...")
        C = np.dot(A, B)
        print("Numpy multiplication done.")
        save_matrix(C, "matC_numpy.txt")
        print("Numpy matrix saved as matC_numpy.txt")
    except Exception as e:
        print(f" Failed to perform numpy test: {e}")
        raise

# Robson hash: b1c9746749e75c15c1c6e398bb77618db177333b3e1e7b6c72540800ebbe956f
# Only exists for temporary testing purposes
def test():
    C1 = read_matrix("matC_mt1.txt")
    print("Shape mt1: ", C1.shape)
    
    C2 = read_matrix("matC_mt3.txt")
    print("Shape mt3: ", C2.shape)
    
    C_np = read_matrix("matC_numpy.txt")
    print("Shape np: ", C_np.shape)
    
    C_st = read_matrix("matC_st.txt")
    print("Shape st: ", C_st.shape)
    
    C_dt = read_matrix("matC_distributed.txt")
    print("Shape dt: ", C_dt.shape)
    
    print("Max difference between matC_mt1 and matC_mt3:", np.max(np.abs(C1 - C2)))
    print("Max difference between matC_mt1 and matC_numpy:", np.max(np.abs(C1 - C_np)))
    print("Max difference between matC_mt1 and matC_st:", np.max(np.abs(C1 - C_st)))
    print("Max difference between matC_mt1 and matC_distributed:", np.max(np.abs(C1 - C_dt)))
    
    print(verify_hash("matC_mt1.txt", "f5531693b6ef5afc011682178243364acb50e57883e6351fb3f73e3b0ae770f7"))
    print(verify_hash("matC_mt3.txt", "f5531693b6ef5afc011682178243364acb50e57883e6351fb3f73e3b0ae770f7"))
    print(verify_hash("matC_numpy.txt", "f5531693b6ef5afc011682178243364acb50e57883e6351fb3f73e3b0ae770f7"))
    print(verify_hash("matC_st.txt", "f5531693b6ef5afc011682178243364acb50e57883e6351fb3f73e3b0ae770f7"))
    print(verify_hash("matC_distributed.txt", "f5531693b6ef5afc011682178243364acb50e57883e6351fb3f73e3b0ae770f7"))

    show_matrix_differences(C1, C2, tol=1e-4)
   
if __name__ == "__main__":
    test()