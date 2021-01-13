
import socket
import pandas as pd
import numpy as np
import tensorflow as tf
import re
import os
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from operator import itemgetter
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import backend as K
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.models import Sequential,  Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, LeakyReLU
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, Callback
from tensorflow.keras import optimizers
from tensorflow.keras.models import load_model
from tensorflow.keras import regularizers
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, roc_auc_score, cohen_kappa_score
from numpy import save,load

# Együttműködik az MT5 Script folder-ben található Getdata script-tel. Ha abban az IsSocketSending=true, akkor a script küldi ennek a megfelelő socket-et

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
        print('connected to', self.addr)
        self.cummdata = ''

        while True:
            data = self.conn.recv(10000)
            self.cummdata+=data.decode("utf-8")
            if not data:
                break    
            self.conn.send(bytes(calcregr(self.cummdata), "utf-8"))
            return self.cummdata
            
    def __del__(self):
        self.sock.close()
        
def calcregr(msg = ''):
    # print(msg)
    msg_parts=msg.split('|')
    columns=msg_parts[0]
    columns = columns.split(',')
    del columns[0]
    date_string=msg_parts[1].split(',')[0]
    features=np.array(msg_parts[1].split(',')[1:])
    data = np.array([features]).astype(np.float)
    # print(date_string)
    # print(columns)
    # print(data)
    df = pd.DataFrame(data, columns=[columns])
    print(df.head(2))
    # chartdata = np.fromstring(msg, dtype=float, sep= ',')
    # Y = np.array(chartdata).reshape(-1,1)
    # X = np.array(np.arange(len(chartdata))).reshape(-1,1)
    #
    # lr = LinearRegression()
    # lr.fit(X, Y)
    # Y_pred = lr.predict(X)
    #
    # P = Y_pred.astype(str).item(-1) + ' ' + Y_pred.astype(str).item(0)
    # print(P)
    return str("OK")
    
serv = socketserver('127.0.0.1', 9090)

while True:  
    msg = serv.recvmsg()

        


    
