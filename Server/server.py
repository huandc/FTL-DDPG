import socket

import pickle
import json
import time


# 建立一个服务端


class Server:
    def __init__(self, addr, port):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.addr = addr
        self.port = port
        self.server.bind((self.addr, self.port))  # 绑定要监听的端口
        self.server.listen(4)
        self.conn = None
        self.server.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, 1)

        self.data = None
        pass

    def wait_connection(self):
        while True:  # conn就是客户端链接过来而在服务端为期生成的一个链接实例
            self.conn, self.addr = self.server.accept()  # 等待链接,多个链接的时候就会出现问题,其实返回了两个值
            print("客户端已连接"+str(self.addr))
            break

    def recv(self):
        total_data = b''
        data = self.conn.recv(1024)
        total_data += data
        num = len(data)
        while len(data) > 0:
            data = self.conn.recv(1024)
            num += len(data)
            total_data += data
            if data[len(data) - 3:len(data)] == b'bye':
                break
        print(str(self.addr)+"接收完成................")
        return pickle.loads(total_data)

    def send(self, data):
        x = pickle.dumps(data)
        # print(type(x))
        self.conn.sendall(x)  # 下载到客户端
        self.conn.sendall(b'bye')  # 下载到客户端

