import socket
from queue import Queue
import time
import threading

import numpy as np


BUFFER_SIZE = 2048

writer = socket.socket()
# writer.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
writer.bind(('localhost', 6969))
writer.listen(1)

time.sleep(3)

print("Iniciando")

cola = Queue()

def enqueue_test():
    arrs = (np.arange(i, 36+i).reshape(-1,4) for i in range(5))
    for arr in arrs:
        time.sleep(2)
        cola.put(arr)

sender = threading.Thread(target=enqueue_test, daemon=True)

sender.start()

# def check_queue():
while True:
    while not cola.empty():
        arr = cola.get()
        b_arr = arr.tobytes()
        sc, address = writer.accept()
        print(f"Conectado: {address}")
        sc.sendall(b_arr)
        print(f"Enviado: {np.frombuffer(b_arr)})")
        sc.close()

    time.sleep(0.2)