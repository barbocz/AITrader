import MetaTrader5 as mt5
import pandas as pd
import socket, numpy as np
# import plotly.graph_objects as go
import matplotlib.pyplot as plt


class socketserver:
    def __init__(self, address='', port=9090):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.address = address
        self.port = port
        self.sock.bind((self.address, self.port))
        self.cummdata = ''

    def recvmsg(self):
        self.sock.listen(1)
        self.conn, self.addr = self.sock.accept()
        print('connected to', self.addr)
        self.cummdata = ''

        while True:
            data = self.conn.recv(10000)
            self.cummdata += data.decode("utf-8")
            if not data:
                break
            self.conn.send(bytes(calcregr(self.cummdata), "utf-8"))
            return self.cummdata

    def __del__(self):
        self.sock.close()


def calcregr(msg=''):
    print(msg)


# connect to MetaTrader 5
if not mt5.initialize():
    print("initialize() failed")
    mt5.shutdown()

# request connection status and parameters
# print(mt5.terminal_info())
# get data on MetaTrader 5 version
# print(mt5.version())

rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_M3, 1, 5000)

mt5.shutdown()
# display each element of obtained data in a new line
# print("Display obtained data 'as is'")
# for rate in rates:
#    print(rate)

# create DataFrame out of the obtained data
rates_frame = pd.DataFrame(rates)

# print(rates_frame)

serv = socketserver('localhost', 9090)

while True:
    msg = serv.recvmsg()