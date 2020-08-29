# 음성 전송 server

import socket
import threading
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import time


class Server:
    def __init__(self):
        self.ip = socket.gethostbyname(socket.gethostname())
        while 1:
            try:
                self.port = 8000

                self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.s.bind((self.ip, self.port))

                break
            except:
                print("Couldn't bind to that port")

        self.connections = []
        self.accept_connections()

    def accept_connections(self):
        self.s.listen(100)

        print('Running on IP: '+self.ip)
        print('Running on port: '+str(self.port))

        while True:
            c, addr = self.s.accept()

            self.connections.append(c)

            threading.Thread(target=self.handle_client,
                             args=(c, addr,)).start()

    def broadcast(self, sock, data):
        for client in self.connections:

            try:

                int_array = np.frombuffer(data, dtype=np.int)
                inverted_int_array = np.invert(int_array)
                int_val = inverted_int_array[0]
                byte_inverted_array = inverted_int_array.tobytes()
                client.send(byte_inverted_array)

            except:
                pass

    def handle_client(self, c, addr):
        while 1:
            try:
                data = c.recv(1024)
                self.broadcast(c, data)

            except socket.error:
                c.close()


server = Server()
