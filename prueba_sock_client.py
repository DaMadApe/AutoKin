from os import read
import socket
from multiprocessing import shared_memory

import numpy as np

BUFFER_SIZE = 2048

reader = socket.socket()
# reader.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
reader.connect(('localhost', 6969))

# while True:
    # sc, address = reader.accept()
    # print(f"Conectado: {address}")

print("Listo")

while True:
    msg = bytearray()
    while True:
        data = reader.recv(BUFFER_SIZE)
        if not data:
            break
        for num in data:
            msg.append(num)
    reader.close()

    arr = np.frombuffer(msg)
    print(f"Recibido {arr}")

    reader = socket.socket()
    # reader.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    reader.connect(('localhost', 6969))