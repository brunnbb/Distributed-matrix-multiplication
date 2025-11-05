# frontend.py
from concurrent import futures
import grpc
import numpy as np
import zlib

import stubs.matrix_pb2 as matrix_pb2
import stubs.matrix_pb2_grpc as matrix_pb2_grpc
from utils import read_matrix, save_matrix, verify_matrix, benchmark, save_stats_json

MAX_MESSAGE_LENGTH = 140 * 1024 * 1024
SERVER_IPS = ["192.168.0.75","192.168.0.65"]
BLOCK_SIZE = 2048
BACKEND_BLOCK_SIZE = 128


class MatrixClient:
    def __init__(self, server_ips, port="50051"):
        self.stubs = [
            matrix_pb2_grpc.MatrixServiceStub(
                grpc.insecure_channel(
                    f"{ip}:{port}",
                    options=[
                        ("grpc.max_send_message_length", MAX_MESSAGE_LENGTH),
                        ("grpc.max_receive_message_length", MAX_MESSAGE_LENGTH),
                    ]
                )
            ) for ip in server_ips
        ]
        self.idx = 0

    def _next_stub(self):
        stub = self.stubs[self.idx]
        self.idx = (self.idx + 1) % len(self.stubs)
        return stub

    def multiply_distributed(self, A, B, block_size=2048, num_servers=1, back_block_size=128, compression=True):
        """
        Split A and B into blocks, send each sub-task to a backend server.
        The backend will queue and process one request at a time using all cores.
        """
        A_blocks = [A[i:i + block_size, :] for i in range(0, A.shape[0], block_size)]
        B_blocks = [B[:, j:j + block_size] for j in range(0, B.shape[1], block_size)]

        futures_list = []
        with futures.ThreadPoolExecutor(max_workers=len(self.stubs)) as executor:
            for i, A_block in enumerate(A_blocks):
                for j, B_block in enumerate(B_blocks):
                    stub = self._next_stub()

                    # Optionally compress large blocks to reduce transfer size
                    if compression:
                        a_bytes = zlib.compress(A_block.tobytes())
                        b_bytes = zlib.compress(B_block.tobytes())
                    else:
                        a_bytes = A_block.tobytes()
                        b_bytes = B_block.tobytes()

                    req = matrix_pb2.MatrixMultiplyRequest(
                        matrix_a=a_bytes,
                        matrix_b=b_bytes,
                        rows_a=A_block.shape[0],
                        cols_a=A_block.shape[1],
                        rows_b=B_block.shape[0],
                        cols_b=B_block.shape[1],
                        block_size=back_block_size,
                        compressed=compression 
                    )

                    future = stub.Multiply.future(req)
                    futures_list.append(((i, j), future))

            partials = {}
            for (i, j), f in futures_list:
                resp = f.result()
                C_block = np.frombuffer(resp.result, dtype=np.float64).reshape((resp.rows, resp.cols))
                partials[(i, j)] = C_block

        # Combine partial results into full matrix
        row_blocks = []
        for i in range(len(A_blocks)):
            row_blocks.append(np.hstack([partials[(i, j)] for j in range(len(B_blocks))]))
        return np.vstack(row_blocks)


def main():
    num_servers = len(SERVER_IPS)
    print(f"Using {num_servers} backend server(s)")

    A = read_matrix("matA.txt")
    B = read_matrix("matB.txt")

    print(f"Matrix A: {A.shape}, Matrix B: {B.shape}")
    client = MatrixClient(SERVER_IPS)

    C, elapsed_time = benchmark(
        "Distributed",
        A, B,
        client.multiply_distributed,
        block_size=BLOCK_SIZE,
        num_servers=num_servers,
        back_block_size=BACKEND_BLOCK_SIZE
    )

    print(f"Done in {elapsed_time:.4f}s")

    if verify_matrix(A, B, C):
        info = {
            "Distributed": {
                "time_seconds": elapsed_time,
                "num_servers": num_servers,
                "block_size": BLOCK_SIZE,
                "backend_block_size": BACKEND_BLOCK_SIZE
            }
        }
        save_stats_json(info)
        save_matrix(C, "matC_distributed.txt")
    else:
        print("Matrix verification failed!")


if __name__ == "__main__":
    print("\nStarting distributed matrix multiplication client...")
    main()
