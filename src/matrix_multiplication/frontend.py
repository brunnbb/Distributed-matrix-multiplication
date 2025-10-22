from concurrent import futures
from math import e
import time

import grpc
import numpy as np

import stubs.matrix_pb2 as matrix_pb2
import stubs.matrix_pb2_grpc as matrix_pb2_grpc

MAX_MESSAGE_LENGTH = 200 * 1024 * 1024

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

        # 1. Divisão da Matriz A
        blocks_a = self._split_matrix(matrix_a)
        
        # Lista para armazenar as chamadas assíncronas
        futures_list = []
        
        # 2. Distribuição das Tarefas
        with futures.ThreadPoolExecutor(max_workers=self.num_servers) as executor:
            for block_a in blocks_a:
                stub = self._get_next_stub()
                
                # Prepara a mensagem gRPC
                request = matrix_pb2.MatrixMultiplyRequest(
                    matrix_a=block_a.flatten().tolist(),
                    rows_a=block_a.shape[0],
                    cols_a=block_a.shape[1],
                    matrix_b=matrix_b.flatten().tolist(),
                    rows_b=matrix_b.shape[0],
                    cols_b=matrix_b.shape[1]
                )
                
                # Executa a chamada gRPC de forma assíncrona
                # O .future() permite que a chamada RPC não bloqueie o ThreadPoolExecutor
                future = stub.Multiply.future(request)
                futures_list.append(future)
                
            # 3. Combinação dos Resultados
            # Aguarda a conclusão de todos os futures
            results = []
            for future in futures_list:
                try:
                    response = future.result()
                    # Reconstrói o bloco de resultado em um array numpy
                    result_block = np.array(response.result).reshape(response.rows, response.cols)
                    results.append(result_block)
                except grpc.RpcError as e:
                    print(f"Erro gRPC em uma chamada: {e}")
                    # Você pode implementar uma lógica de retry aqui se necessário
                    return None
            
            # Concatena os blocos verticalmente (por linhas)
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
    MATRIX_SIZE = 4096
    
    np.random.seed(42) 
    A = np.random.rand(MATRIX_SIZE, MATRIX_SIZE)
    B = np.random.rand(MATRIX_SIZE, MATRIX_SIZE)
    
    client = MatrixClient(SERVER_IPS)
    
    print(f"\nMatrix A: {A.shape}, Matrix B: {B.shape}")
    print(f"Distributing the multiplication in {NUM_SERVERS} servers...\n")

    start_time = time.perf_counter()
    result_distributed = client.multiply_distributed(A, B)
    end_time = time.perf_counter()
    print(f"Distributed multiplication time: {end_time - start_time:.4f} seconds")
    
    
    if result_distributed is not None:
        print("\n--- Verification with Numpy.dot (Local) ---")
        result_local = np.dot(A, B)        
        print(f"Result from distributed multiplication: {result_distributed[0:3,0:3]}")
        print(f"Result from numpy.dot: {result_local[0:3,0:3]}")
        is_close = np.allclose(result_distributed, result_local)
        print(f"The distributed results are close to numpy.dot? {is_close}")
    else:
        print("\nThe multiplication could not be completed due to errors in gRPC calls.")

if __name__ == '__main__':
    main()