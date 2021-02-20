import configparser
import os
import sys


metatrader_dir=''
ml_type='CNN'
symbol=''
model_name=''
socket_port=9090
model_path=''

def set():

    global metatrader_dir
    global ml_type
    global symbol
    global model_name
    global socket_port
    global model_path
    cfg = configparser.ConfigParser()

    if (len(sys.argv)==1):
        cfg.read(os.path.join('..','..','project_default.ini'))
    else:
        cfg.read(os.path.join('..', '..', sys.argv[1]+'.ini'))

    ml_type=cfg['parameters']['ml_type']
    symbol = cfg['parameters']['symbol']
    model_name = cfg['parameters']['model_name']
    metatrader_dir = cfg['parameters']['metatrader_dir'] + 'AI\\' + ml_type + '\\' + symbol + '\\' + model_name + '\\'
    socket_port=cfg['parameters']['socket_port']
    model_path='best_models\\'+symbol+'_'+model_name

