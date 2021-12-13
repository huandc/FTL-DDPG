import ast
import socket
import demjson
import json
from .NumpyEncoder import NumpyEncoder
import pickle
import time


# 建立一个服务端


class Client:
    def __init__(self, addr, port):
        self.conn = socket.socket()  # 等待链接,多个链接的时候就会出现问题,其实返回了两个值
        self.addr = addr
        self.port = port
        self.data = None
        pass

    def wait_connection(self):
        while True:  # conn就是客户端链接过来而在服务端为期生成的一个链接实例
            self.conn.connect((self.addr, self.port))
            print("已连接")
            break

    def recv(self):
        total_data = b''
        data = self.conn.recv(1024)
        total_data += data
        num = len(data)
        # 我一开始以为如果没有数据了，读出来的data长度为0，len(data)==0，从而导致卡在while循环中
        while len(data) > 0:
            data = self.conn.recv(1024)
            num += len(data)
            # print(str(len(data)) + "  " + str(num))
            total_data += data
            # print(data[len(data) - 3:len(data)])
            if data[len(data) - 3:len(data)] == b'bye':
                break
            # print(len(data))

        return pickle.loads(total_data)

    def send(self, data):

        x = pickle.dumps(data)
        # print(type(x))
        self.conn.sendall(x)  # 下载到客户端
        self.conn.sendall(b'bye')  # 下载到客户端
