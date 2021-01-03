# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 01:52:04 2019

@author: dmitrievsky
"""
import socket, numpy as np
from sklearn.linear_model import LinearRegression

class socketserver:
    def __init__(self, address = 'localhost', port = 9090):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.address = address
        self.port = port
        self.sock.bind((self.address, self.port))
        self.cummdata = ''
        
    def recvmsg(self):
        self.sock.listen(1)
        self.conn, self.addr = self.sock.accept()
        # print('connected to', self.addr)
        self.cummdata = ''

        while True:
            data = self.conn.recv(1000000)
            self.cummdata+=data.decode("utf-8")
            if not data:
                break    
            self.conn.send(bytes(calcregr(self.cummdata), "utf-8"))
            return self.cummdata
            
    def __del__(self):
        self.sock.close()
        
def calcregr(msg = ''):
    # print(msg)
    items = msg.split(' ')
    print(len(items))
    return str('back')
    
serv = socketserver('127.0.0.1', 9090)

while True:  
    msg = serv.recvmsg()

        
        

    
