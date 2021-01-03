import os
import re
import pandas as pd
f = open("C:\\Users\\Barbocz Attila\\AppData\\Roaming\\MetaQuotes\\Terminal\\67381DD86A2959850232C0BA725E5966\\MQL5\\Files\\Parameters.txt", "r")
model_path=re.sub('[^A-Za-z0-9]+', '', f.readline().split(':')[1])
best_model_path = os.path.join('.', 'best_models',model_path)
try:
    os.mkdir(best_model_path)
except OSError:
    pass
else:
    print ("Successfully created the directory %s " % best_model_path)
print(best_model_path)
# os.system("copy  C:\\Users\\BARBOC~1\\AppData\\Roaming\\MetaQuotes\\Terminal\\67381DD86A2959850232C0BA725E5966\\MQL5\\Files\\*.* "+best_model_path)

# os.system("copy  C:\\Users\\BARBOC~1\\install.ini c:\\Temp\\data")

# C:\Users\Barbocz Attila\AppData\Roaming\MetaQuotes\Terminal\67381DD86A2959850232C0BA725E5966\MQL5\Files
#
# df=pd.read_csv("C:\\Users\\Barbocz Attila\\AppData\\Roaming\\MetaQuotes\\Terminal\\67381DD86A2959850232C0BA725E5966\\MQL5\\Files\\Training.csv")
# df=df.drop(columns=['date','open','high','low','close'])
# pd.options.display.max_columns = 270
# print(df.describe())