#!/usr/bin/env python3
import time
import platform
import numpy as np
from numba import njit, prange
import os
from datetime import datetime

# -----------------------------
# Integer benchmark (MIPS)
# -----------------------------
@njit(fastmath=True)
def int_ops(iters):
    a = 1
    b = 2
    c = 3
    for i in range(iters):
        a = a + b
        b = b ^ c
        c = c + a
        a = a - 1
    return a + b + c

# -----------------------------
# Floating benchmark (FLOPS)
# -----------------------------
@njit(fastmath=True)
def float_ops(iters):
    x = 1.1
    y = 2.2
    z = 3.3
    for i in range(iters):
        # 4 flops: 2 mul + 2 add
        x = x * y + z
        y = y * x + z
    return x + y + z 


# -----------------------------
# Parallel versions 
# -----------------------------
@njit(parallel=True, fastmath=True)
def int_ops_parallel(iters_per_thread):
    acc = 0
    for i in prange(iters_per_thread):
        acc += int_ops(10_000)
    return acc


@njit(parallel=True, fastmath=True)
def float_ops_parallel(iters_per_thread):
    acc = 0.0
    for i in prange(iters_per_thread):
        acc += float_ops(10_000)
    return acc


# -----------------------------
# Benchmark runner
# -----------------------------
def benchmark(iters=50_000_000, parallel=False):
    if parallel:
        print("\n⚙️  Running multi-core benchmark (Numba parallel)...")
    else:
        print("\n⚙️  Running single-core benchmark...")

    # Warm-up compilation
    print("Compiling Numba functions (JIT warm-up)...")
    int_ops(10)
    float_ops(10)
    if parallel:
        int_ops_parallel(1)
        float_ops_parallel(1)

    print("\n🧮 Measuring integer performance (MIPS)...")
    t0 = time.perf_counter()
    if parallel:
        int_ops_parallel(iters // 10_000)
    else:
        int_ops(iters)
    t1 = time.perf_counter()
    elapsed = t1 - t0
    ops_per_iter = 4
    ips = iters * ops_per_iter / elapsed
    mips = ips / 1e6
    print(f"Elapsed: {elapsed:.3f}s")
    print(f"MIPS ≈ {mips:,.2f}")

    print("\n🔢 Measuring floating-point performance (FLOPS)...")
    t0 = time.perf_counter()
    if parallel:
        float_ops_parallel(iters // 10_000)
    else:
        float_ops(iters)
    t1 = time.perf_counter()
    elapsed = t1 - t0
    flops_per_iter = 4
    flops = iters * flops_per_iter / elapsed
    print(f"Elapsed: {elapsed:.3f}s")
    print(f"FLOPS ≈ {flops:,.2f}  (~{flops / 1e9:.2f} GFLOPS)")

    print("\n✅ Done.")
    return mips, flops


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sys_info = f"\nSystem: {platform.platform()}\nCPU cores: {os.cpu_count()}\nTimestamp: {now}"
    print(sys_info)

    # --- Single-thread test ---
    mips, flops = benchmark(iters=50_000_000, parallel=False)

    # --- Multi-thread test ---
    print("\n--------------------------------")
    print("Now running parallel test...")
    print("--------------------------------")
    mips_p, flops_p = benchmark(iters=50_000_000, parallel=True)

    output = f"""
=========================================
Numba Hardware Benchmark Results
=========================================
{sys_info}

Single-core:
  MIPS:  {mips:,.2f}
  FLOPS: {flops:,.2f}  (~{flops/1e9:.2f} GFLOPS)

Multi-core:
  MIPS:  {mips_p:,.2f}
  FLOPS: {flops_p:,.2f}  (~{flops_p/1e9:.2f} GFLOPS)

-----------------------------------------
"""
    with open(r"src/matrix_multiplication/logs/benchmark_results.log", "w", encoding="utf-8") as f:
        f.write(output.strip())

    print("\n✅ Results saved to 'benchmark_results.log'")
