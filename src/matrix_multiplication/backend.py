from concurrent import futures
import time

import grpc
import numpy as np

import stubs.matrix_pb2 as matrix_pb2
import stubs.matrix_pb2_grpc as matrix_pb2_grpc
from mt_local import block_matrix_multiply

MAX_MESSAGE_LENGTH = 140 * 1024 * 1024

class MatrixBackend(matrix_pb2_grpc.MatrixServiceServicer):

    def Multiply(self, request, context):
        # Reconstruct matrices from bytes
        A = np.frombuffer(request.matrix_a, dtype=np.float64).reshape((request.rows_a, request.cols_a))
        B = np.frombuffer(request.matrix_b, dtype=np.float64).reshape((request.rows_b, request.cols_b))
        num_cores = request.num_cores
        block_size = request.block_size

        print(f"Received matrices for multiplication: A{A.shape} x B{B.shape}")
        print(f"Using {num_cores} cores with block size {block_size}...")

        # Perform matrix multiplication
        start_time = time.perf_counter()
        result = block_matrix_multiply(A, B, block_size=block_size, num_cores=num_cores)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        
        print(f"Done in {elapsed_time:.4f} seconds. Result shape: {result.shape}")
        
        # Return the result as a flattened array
        return matrix_pb2.MatrixMultiplyResponse(
            result=result.tobytes(),
            rows=result.shape[0],
            cols=result.shape[1]
        )

def serve():
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ("grpc.max_send_message_length", MAX_MESSAGE_LENGTH),
            ("grpc.max_receive_message_length", MAX_MESSAGE_LENGTH),
        ]
    )
    matrix_pb2_grpc.add_MatrixServiceServicer_to_server(MatrixBackend(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Backend server is running on port 50051...")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
