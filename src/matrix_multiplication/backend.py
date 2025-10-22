from concurrent import futures

import grpc
import numpy as np

import stubs.matrix_pb2 as matrix_pb2
import stubs.matrix_pb2_grpc as matrix_pb2_grpc
from mt_local import block_matrix_multiply

MAX_MESSAGE_LENGTH = 200 * 1024 * 1024

class MatrixBackend(matrix_pb2_grpc.MatrixServiceServicer):

    def Multiply(self, request, context):
        # Deserialize matrix blocks
        matrix_a = np.array(request.matrix_a).reshape(request.rows_a, request.cols_a)
        matrix_b = np.array(request.matrix_b).reshape(request.rows_b, request.cols_b)

        print(f"Received matrices for multiplication: A{matrix_a.shape} x B{matrix_b.shape}")

        # Perform matrix multiplication
        result = block_matrix_multiply(matrix_a, matrix_b, block_size=256, num_cores=10)
        
        print(f"Completed multiplication. Result shape: {result.shape}")
        
        # Return the result as a flattened array
        return matrix_pb2.MatrixMultiplyResponse(
            result=result.flatten().tolist(),
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
