import numpy as np
from numba import njit

from utils import read_matrix, save_matrix, benchmark, verify_matrix

@njit()
def mult(A, B):
    C = np.zeros((len(A), len(B[0])))
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                C[i, j] += A[i, k] * B[k, j]  
    return C

def main():
    A = read_matrix("matA.txt")
    B = read_matrix("matB.txt")
    C = benchmark("Single-threaded", A, B, mult)
    print("Benchmarking complete.")    
    verify_matrix(A, B, C)
    save_matrix(C, "matC_st.txt")

if __name__ == "__main__":
    print("Running...")
    main()
    print("Done!")
        