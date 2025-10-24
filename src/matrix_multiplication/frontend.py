from concurrent import futures
import time

import grpc
import numpy as np

import stubs.matrix_pb2 as matrix_pb2
import stubs.matrix_pb2_grpc as matrix_pb2_grpc

from utils import read_matrix, save_matrix, benchmark, verify_matrix, truncate_matrix

MAX_MESSAGE_LENGTH = 280 * 1024 * 1024

class MatrixClient:

    def __init__(self, server_ips, port='50051'):
        if not server_ips:
            raise ValueError("A lista de IPs dos servidores não pode estar vazia.")    
        self.servers = [f"{ip}:{port}" for ip in server_ips]
        self.num_servers = len(self.servers)
        self.stubs = []
        for server_address in self.servers:
            channel = grpc.insecure_channel(
                server_address, 
                options=[
                    ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
                    ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                ]
            )
            stub = matrix_pb2_grpc.MatrixServiceStub(channel)
            self.stubs.append(stub)
            print(f"Canal e Stub criados para: {server_address}")
        self.server_index = 0

    def _get_next_stub(self):
        stub = self.stubs[self.server_index]
        self.server_index = (self.server_index + 1) % self.num_servers
        return stub

    def _split_matrix(self, matrix_a):
        """
        Divide a Matriz A por linhas para distribuição.
        A Matriz B é enviada inteira para todos (assumindo que B é menor 
        ou a divisão ideal é só nas linhas de A para a multiplicação M x N).
        """
        rows_a = matrix_a.shape[0]
        rows_per_server = rows_a // self.num_servers
        remainder_rows = rows_a % self.num_servers
        
        blocks_a = []
        start_row = 0
        for i in range(self.num_servers):
            # Distribui as linhas restantes uniformemente
            end_row = start_row + rows_per_server + (1 if i < remainder_rows else 0)
            blocks_a.append(matrix_a[start_row:end_row, :])
            start_row = end_row
        return blocks_a

    def multiply_distributed(self, matrix_a, matrix_b):
        if matrix_a.shape[1] != matrix_b.shape[0]:
            raise ValueError("As dimensões das matrizes não são compatíveis para multiplicação.")

        blocks_a = self._split_matrix(matrix_a)
        
        # Lista para armazenar as chamadas assíncronas
        futures_list = []
        
        # Distribuição das Tarefas
        with futures.ThreadPoolExecutor(max_workers=self.num_servers) as executor:
            for block_a in blocks_a:
                stub = self._get_next_stub()
                
                # Prepara a mensagem gRPC
                request = matrix_pb2.MatrixMultiplyRequest(
                    matrix_a=block_a.tobytes(),
                    rows_a=block_a.shape[0],
                    cols_a=block_a.shape[1],
                    matrix_b=matrix_b.tobytes(),
                    rows_b=matrix_b.shape[0],
                    cols_b=matrix_b.shape[1]
                )
                
                # Executa a chamada gRPC de forma assíncrona
                # O .future() permite que a chamada RPC não bloqueie o ThreadPoolExecutor
                future = stub.Multiply.future(request)
                futures_list.append(future)
                
            # Aguarda a conclusão de todos os futures para combinar os resultados
            results = []
            for future in futures_list:
                try:
                    response = future.result()
                    # Reconstrói o bloco de resultado em um array numpy
                    result_block = np.frombuffer(response.result, dtype=np.float64).reshape((response.rows, response.cols))
                    results.append(result_block)
                except grpc.RpcError as e:
                    print(f"Erro gRPC em uma chamada: {e}")
                    # TODO: Lidar com falhas de servidor ou reintentar a chamada
                    return None
            
            # Concatena os blocos verticalmente 
            if results:
                final_result = np.vstack(results)
                return final_result
            else:
                return None

def main():
    SERVER_IPS = [
        "192.168.0.75",
    ]
    NUM_SERVERS = len(SERVER_IPS)
    
    A: np.ndarray = read_matrix("matA.txt")
    B: np.ndarray = read_matrix("matB.txt")
    
    print(f"\nMatrix A: {A.shape}, Matrix B: {B.shape}")
    print(f"Distributing the multiplication in {NUM_SERVERS} servers...\n")

    client = MatrixClient(SERVER_IPS)
    result_distributed = benchmark("Distributed", A, B, client.multiply_distributed)
    verify_matrix(A, B, result_distributed)
    save_matrix(result_distributed, "matC_distributed.txt")
    

if __name__ == '__main__':
    print("Running Frontend...")
    main()
    print("Frontend finished.")