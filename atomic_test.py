import os
import re
import pandas as pd
import pandas_profiling

# f = open("C:\\Users\\Barbocz Attila\\AppData\\Roaming\\MetaQuotes\\Terminal\\67381DD86A2959850232C0BA725E5966\\MQL5\\Files\\Parameters.txt", "r")
# # print(f.readline().split(':')[1])
# # f.readline().split(':')[1]
# model_path=re.sub('[^A-Za-z0-9]+', '', f.readline().split(':')[1])
# best_model_path = os.path.join('.', 'ai','cnn','best_models','GBPUSD1','columns_needed.csv')
# print(best_model_path)
# os.system('copy source.txt destination.txt')

df=pd.read_csv("C:\\Users\\Barbocz Attila\\AppData\\Roaming\\MetaQuotes\\Terminal\\67381DD86A2959850232C0BA725E5966\\MQL5\\Files\\Training.csv")
# df=pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/planets.csv')
print(df.profile_report())
