import _thread
import numpy as np
from server import Server
from CountDownLauch import CountDownLatch

epoch = 0
c = CountDownLatch(2)

para1 = None
para2 = None

#
# def process(list):
#     print("服务器 数据Aggregation")
#
#     data = np.mean(np.array([list[0], list[1]]), axis=0)
#
#     print("数据Aggregation 完成")
#
#     return data
#
#
# list = []


def get1(server):
    global para1
    para1 = server.recv()
    c.countDown()


def get2(server):
    global para2
    para2 = server.recv()
    c.countDown()


def send(server, data):
    server.send(data)


print("服务器已启动............")
server1 = Server('127.0.0.1', 7070)
client1 = Server('127.0.0.1', 7071)
client2 = Server('127.0.0.1', 7072)
# client3 = Server('127.0.0.1', 7073)

server1.wait_connection()
client1.wait_connection()
client2.wait_connection()
# client3.wait_connection()
print("全部已经连接")
while True:
    epoch = epoch + 1
    print("*" * 8 + str(epoch) + "次等待开始" + "*" * 8)
    _thread.start_new_thread(get1, (client1,))
    _thread.start_new_thread(get2, (client2,))
    # _thread.start_new_thread(get, (client3,))

    c.wait()
    data = np.mean(np.array([para1, para2]), axis=0)
    print(str(epoch) + "服务器 参数下发到全局moedl")
    server1.send(data)

    print(str(epoch) + "等待server模型训练完成传递参数")
    server_data = server1.recv()

    print(str(epoch) + "收到参数, 开始下发.............")
    data = np.mean(np.array([server_data, para1]), axis=0)
    client1.send(data)
    data = np.mean(np.array([server_data, para2]), axis=0)
    client2.send(data)
    # client3.send(data)
    print(str(epoch) + "参数下发完成.....")
    # list.clear()
    c = CountDownLatch(2)
