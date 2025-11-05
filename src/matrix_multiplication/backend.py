from concurrent import futures
import threading
import queue
import time
import os
import zlib

import grpc
import numpy as np

import stubs.matrix_pb2 as matrix_pb2
import stubs.matrix_pb2_grpc as matrix_pb2_grpc
from mt_local import block_matrix_multiply

# Allow larger gRPC messages
MAX_MESSAGE_LENGTH = 140 * 1024 * 1024
GRPC_MAX_WORKERS = 4 

# Single job queue for sequential heavy processing
_job_queue = queue.Queue()

class _Job:
    """Encapsulates a queued job and synchronization with the gRPC handler."""
    def __init__(self, request, context):
        self.request = request
        self.context = context
        self.done = threading.Event()
        self.response = None
        self.exception = None


class MatrixBackend(matrix_pb2_grpc.MatrixServiceServicer):
    def Multiply(self, request, context):
        """
        Enqueue request and block until it completes.
        Ensures only one heavy multiplication runs at a time (using all CPU cores).
        """
        job = _Job(request, context)
        _job_queue.put(job)

        print("Request added to queue. Waiting for worker...")

        # Wait for the worker to finish this job
        job.done.wait()

        if job.exception:
            context.set_details(str(job.exception))
            context.set_code(grpc.StatusCode.INTERNAL)
            return matrix_pb2.MatrixMultiplyResponse(result=b"", rows=0, cols=0)

        return job.response
    

def _worker_loop():
    """Continuously consume jobs from the queue and run them using all CPU cores."""
    while True:
        job = _job_queue.get()
        if job is None:
            break  # Sentinel for shutdown

        req = job.request
        try:
            # Decompress if flag is set
            a_bytes = req.matrix_a
            b_bytes = req.matrix_b
            if getattr(req, "compressed", False):
                a_bytes = zlib.decompress(a_bytes)
                b_bytes = zlib.decompress(b_bytes)

            # Reconstruct numpy matrices
            A = np.frombuffer(a_bytes, dtype=np.float64).reshape((req.rows_a, req.cols_a))
            B = np.frombuffer(b_bytes, dtype=np.float64).reshape((req.rows_b, req.cols_b))

            num_cores = req.num_cores if req.num_cores < os.cpu_count() else os.cpu_count()
            block_size = getattr(req, "block_size", 128)

            print(f"[worker] Processing job: A{A.shape} x B{B.shape}, "
                  f"using {num_cores} cores, block_size={block_size}")

            start = time.perf_counter()
            result = block_matrix_multiply(A, B, block_size=block_size, num_cores=num_cores)
            elapsed = time.perf_counter() - start

            print(f"[worker] ✅ Done in {elapsed:.4f}s — result shape {result.shape}")

            # Build response (no compression on response for simplicity)
            job.response = matrix_pb2.MatrixMultiplyResponse(
                result=result.tobytes(),
                rows=result.shape[0],
                cols=result.shape[1],
            )

        except Exception as e:
            job.exception = e
        finally:
            job.done.set()
            _job_queue.task_done()


# Launch background worker thread
_worker_thread = threading.Thread(target=_worker_loop, daemon=True)
_worker_thread.start()    

def serve(port="50051"):
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=GRPC_MAX_WORKERS),
        options=[
            ("grpc.max_send_message_length", MAX_MESSAGE_LENGTH),
            ("grpc.max_receive_message_length", MAX_MESSAGE_LENGTH),
        ],
    )
    matrix_pb2_grpc.add_MatrixServiceServicer_to_server(MatrixBackend(), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"Backend running on port {port}...")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
