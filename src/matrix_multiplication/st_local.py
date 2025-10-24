import numpy as np
from numba import njit

from utils import read_matrix, save_matrix, benchmark, verify_matrix

@njit()
def mult(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    C = np.zeros((A.shape[0], B.shape[1]), dtype=np.float64)
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(A.shape[1]):
                C[i, j] += A[i, k] * B[k, j]  
    return C

def main():
    A: np.ndarray = read_matrix("matA.txt")
    B: np.ndarray = read_matrix("matB.txt")
    C: np.ndarray = benchmark("Single-threaded", A, B, mult)
    print("Benchmarking complete.")    
    verify_matrix(A, B, C)
    save_matrix(C, "matC_st.txt")

if __name__ == "__main__":
    print("Running...")
    main()
    print("Done!")
        