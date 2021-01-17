import os
import re
import pandas as pd,numpy as np
metatrader_dir="C:\\Users\\Barbocz Attila\\AppData\\Roaming\\MetaQuotes\\Terminal\\67381DD86A2959850232C0BA725E5966\\MQL5\Files\\"
# f = open(metatrader_dir+"Parameters.txt", "r")
# model_path=re.sub('[^A-Za-z0-9]+', '', f.readline().split(':')[1])
# best_model_path = os.path.join('.', 'best_models',model_path)
# try:
#     os.mkdir(best_model_path)
# except OSError:
#     pass
# else:
#     print ("Successfully created the directory %s " % best_model_path)
# print(best_model_path)


# df=pd.read_csv(metatrader_dir+"Training.csv")
# df=df.drop(columns=['date','open','high','low'])
# print(len(df.columns[df.nunique() <= 1]))
# df = df.drop(columns=df.columns[df.nunique() <= 1])
# pd.options.display.max_columns = 270
# print(df.describe())
# df=pd.read_csv("best_models/USDCHF1/output_new.csv")
# pd.set_option('display.max_rows', 200)
# # print(df.iloc[0:8,2]==2)
# print(df[(df.iloc[:,2]==1) & (df.iloc[:,1]>0.80)].head(200))

# metatrader_dir="C:\\Users\\melgibson\\AppData\Roaming\\MetaQuotes\\Terminal\\6E837615CE50F086D7E2801AA8E2160A\\MQL5\\Files\\"
# df=pd.read_csv("output_new.csv")
# df.to_csv(metatrader_dir+'output_new.csv',header=False,index=False)
# os.system("copy  C:\\Users\\BARBOC~1\\AppData\\Roaming\\MetaQuotes\\Terminal\\67381DD86A2959850232C0BA725E5966\\MQL5\\Files\\*.* "+best_model_path)
# s_data="2021.01.13 16:06,1.21657,1.21672,1.21638,1.21654,276,57.41245"
# s_cols="date,open,high,low,close,volume,mfi"
# s_cols=s_cols.split(',')
# del s_cols[0]
# print(s_cols)
# data=(s_data.split(',')[1:])
# data=np.array([data]).astype(np.float)
# # data = np.array([[2014,"toyota","corolla"],
# #                  [2018,"honda","civic"],
# #                  [2020,"hyndai","accent"],
# #                  [2017,"nissan","sentra"]])
# # print(data)
#
# # data = [['tom', 10], ['nick', 15], ['juli', 14]]
# #
# # # Create the pandas DataFrame
# # df = pd.DataFrame(data, columns=["open,high,low,close,volume,mfi(5)"])
# df = pd.DataFrame(data,columns = [s_cols])
# # df.loc[0] = data
# #
# print(df.head(2))

#
# s_data="1.21657,1.21672,1.21638"
# s_cols="open,high,low"
# s_cols=s_cols.split(',')
# print(s_cols)
# data=(s_data.split(',')[0:])
# data=np.array([data]).astype(np.float)
#
# # data = np.array([[2014,"toyota","corolla"],
# #                  [2018,"honda","civic"],
# #                  [2020,"hyndai","accent"],
# #                  [2017,"nissan","sentra"]])
# print(data)
# # pass column names in the columns parameter
# df = pd.DataFrame(data, columns = s_cols)
# print(df.head(4))

# example of a standardization


# msg="date,open,high,low,close,volume,mfi(5)_T3_B0,mfi(5)_T4_B0,mfi(5)_T6_B0,mfi(5)_T10_B0,mfi(5)_T15_B0,mfi(5)_T30_B0,mfi(5)_T60_B0,laguerre_rsi_wi_T3_B0,laguerre_rsi_wi_T4_B0,laguerre_rsi_wi_T6_B0,laguerre_rsi_wi_T10_B0,laguerre_rsi_wi_T15_B0,laguerre_rsi_wi_T30_B0,laguerre_rsi_wi_T60_B0,laguerre_rsi_wi_T3_B1,laguerre_rsi_wi_T4_B1,laguerre_rsi_wi_T6_B1,laguerre_rsi_wi_T10_B1,laguerre_rsi_wi_T15_B1,laguerre_rsi_wi_T30_B1,laguerre_rsi_wi_T60_B1,laguerre_rsi_wi_T3_B2,laguerre_rsi_wi_T4_B2,laguerre_rsi_wi_T6_B2,laguerre_rsi_wi_T10_B2,laguerre_rsi_wi_T15_B2,laguerre_rsi_wi_T30_B2,laguerre_rsi_wi_T60_B2,laguerre_rsi_wi_T3_B3,laguerre_rsi_wi_T4_B3,laguerre_rsi_wi_T6_B3,laguerre_rsi_wi_T10_B3,laguerre_rsi_wi_T15_B3,laguerre_rsi_wi_T30_B3,laguerre_rsi_wi_T60_B3,laguerre_rsi_wi_T3_B4,laguerre_rsi_wi_T4_B4,laguerre_rsi_wi_T6_B4,laguerre_rsi_wi_T10_B4,laguerre_rsi_wi_T15_B4,laguerre_rsi_wi_T30_B4,laguerre_rsi_wi_T60_B4,3_minutes_relat_T3_B0,3_minutes_relat_T4_B0,3_minutes_relat_T6_B0,3_minutes_relat_T10_B0,3_minutes_relat_T15_B0,3_minutes_relat_T30_B0,3_minutes_relat_T60_B0,3_minutes_relat_T3_B1,3_minutes_relat_T4_B1,3_minutes_relat_T6_B1,3_minutes_relat_T10_B1,3_minutes_relat_T15_B1,3_minutes_relat_T30_B1,3_minutes_relat_T60_B1,3_minutes_relat_T3_B2,3_minutes_relat_T4_B2,3_minutes_relat_T6_B2,3_minutes_relat_T10_B2,3_minutes_relat_T15_B2,3_minutes_relat_T30_B2,3_minutes_relat_T60_B2,3_minutes_relat_T3_B3,3_minutes_relat_T4_B3,3_minutes_relat_T6_B3,3_minutes_relat_T10_B3,3_minutes_relat_T15_B3,3_minutes_relat_T30_B3,3_minutes_relat_T60_B3,3_minutes_relat_T3_B4,3_minutes_relat_T4_B4,3_minutes_relat_T6_B4,3_minutes_relat_T10_B4,3_minutes_relat_T15_B4,3_minutes_relat_T30_B4,3_minutes_relat_T60_B4,3_minutes_relat_T3_B5,3_minutes_relat_T4_B5,3_minutes_relat_T6_B5,3_minutes_relat_T10_B5,3_minutes_relat_T15_B5,3_minutes_relat_T30_B5,3_minutes_relat_T60_B5,3_minutes_relat_T3_B6,3_minutes_relat_T4_B6,3_minutes_relat_T6_B6,3_minutes_relat_T10_B6,3_minutes_relat_T15_B6,3_minutes_relat_T30_B6,3_minutes_relat_T60_B6,3_minutes_relat_T3_B7,3_minutes_relat_T4_B7,3_minutes_relat_T6_B7,3_minutes_relat_T10_B7,3_minutes_relat_T15_B7,3_minutes_relat_T30_B7,3_minutes_relat_T60_B7,3_minutes_relat_T3_B8,3_minutes_relat_T4_B8,3_minutes_relat_T6_B8,3_minutes_relat_T10_B8,3_minutes_relat_T15_B8,3_minutes_relat_T30_B8,3_minutes_relat_T60_B8,fxc_activityfor_T3_B0,fxc_activityfor_T4_B0,fxc_activityfor_T6_B0,fxc_activityfor_T10_B0,fxc_activityfor_T15_B0,fxc_activityfor_T30_B0,fxc_activityfor_T60_B0,fxc_activityfor_T3_B1,fxc_activityfor_T4_B1,fxc_activityfor_T6_B1,fxc_activityfor_T10_B1,fxc_activityfor_T15_B1,fxc_activityfor_T30_B1,fxc_activityfor_T60_B1,fxc_activityfor_T3_B2,fxc_activityfor_T4_B2,fxc_activityfor_T6_B2,fxc_activityfor_T10_B2,fxc_activityfor_T15_B2,fxc_activityfor_T30_B2,fxc_activityfor_T60_B2,fxc_activityfor_T3_B3,fxc_activityfor_T4_B3,fxc_activityfor_T6_B3,fxc_activityfor_T10_B3,fxc_activityfor_T15_B3,fxc_activityfor_T30_B3,fxc_activityfor_T60_B3,fxc_activityfor_T3_B4,fxc_activityfor_T4_B4,fxc_activityfor_T6_B4,fxc_activityfor_T10_B4,fxc_activityfor_T15_B4,fxc_activityfor_T30_B4,fxc_activityfor_T60_B4,fxc_activityfor_T3_B5,fxc_activityfor_T4_B5,fxc_activityfor_T6_B5,fxc_activityfor_T10_B5,fxc_activityfor_T15_B5,fxc_activityfor_T30_B5,fxc_activityfor_T60_B5,rsi(14)_T3_B0,rsi(14)_T4_B0,rsi(14)_T6_B0,rsi(14)_T10_B0,rsi(14)_T15_B0,rsi(14)_T30_B0,rsi(14)_T60_B0,waddah_attar_ex_T3_B0,waddah_attar_ex_T4_B0,waddah_attar_ex_T6_B0,waddah_attar_ex_T10_B0,waddah_attar_ex_T15_B0,waddah_attar_ex_T30_B0,waddah_attar_ex_T60_B0,waddah_attar_ex_T3_B1,waddah_attar_ex_T4_B1,waddah_attar_ex_T6_B1,waddah_attar_ex_T10_B1,waddah_attar_ex_T15_B1,waddah_attar_ex_T30_B1,waddah_attar_ex_T60_B1,waddah_attar_ex_T3_B2,waddah_attar_ex_T4_B2,waddah_attar_ex_T6_B2,waddah_attar_ex_T10_B2,waddah_attar_ex_T15_B2,waddah_attar_ex_T30_B2,waddah_attar_ex_T60_B2,waddah_attar_ex_T3_B3,waddah_attar_ex_T4_B3,waddah_attar_ex_T6_B3,waddah_attar_ex_T10_B3,waddah_attar_ex_T15_B3,waddah_attar_ex_T30_B3,waddah_attar_ex_T60_B3,macd(12_26_9)_T3_B0,macd(12_26_9)_T4_B0,macd(12_26_9)_T6_B0,macd(12_26_9)_T10_B0,macd(12_26_9)_T15_B0,macd(12_26_9)_T30_B0,macd(12_26_9)_T60_B0,macd(12_26_9)_T3_B1,macd(12_26_9)_T4_B1,macd(12_26_9)_T6_B1,macd(12_26_9)_T10_B1,macd(12_26_9)_T15_B1,macd(12_26_9)_T30_B1,macd(12_26_9)_T60_B1,stoch(8_5_3)_T3_B0,stoch(8_5_3)_T4_B0,stoch(8_5_3)_T6_B0,stoch(8_5_3)_T10_B0,stoch(8_5_3)_T15_B0,stoch(8_5_3)_T30_B0,stoch(8_5_3)_T60_B0,stoch(8_5_3)_T3_B1,stoch(8_5_3)_T4_B1,stoch(8_5_3)_T6_B1,stoch(8_5_3)_T10_B1,stoch(8_5_3)_T15_B1,stoch(8_5_3)_T30_B1,stoch(8_5_3)_T60_B1,murreys_math_os_T3_B0,murreys_math_os_T4_B0,murreys_math_os_T6_B0,murreys_math_os_T10_B0,murreys_math_os_T15_B0,murreys_math_os_T30_B0,murreys_math_os_T60_B0,murreys_math_os_T3_B1,murreys_math_os_T4_B1,murreys_math_os_T6_B1,murreys_math_os_T10_B1,murreys_math_os_T15_B1,murreys_math_os_T30_B1,murreys_math_os_T60_B1,sonicr_pva_volu_T3_B0,sonicr_pva_volu_T4_B0,sonicr_pva_volu_T6_B0,sonicr_pva_volu_T10_B0,sonicr_pva_volu_T15_B0,sonicr_pva_volu_T30_B0,sonicr_pva_volu_T60_B0,sonicr_pva_volu_T3_B1,sonicr_pva_volu_T4_B1,sonicr_pva_volu_T6_B1,sonicr_pva_volu_T10_B1,sonicr_pva_volu_T15_B1,sonicr_pva_volu_T30_B1,sonicr_pva_volu_T60_B1,sonicr_pva_volu_T3_B2,sonicr_pva_volu_T4_B2,sonicr_pva_volu_T6_B2,sonicr_pva_volu_T10_B2,sonicr_pva_volu_T15_B2,sonicr_pva_volu_T30_B2,sonicr_pva_volu_T60_B2|2021.01.13 15:33,1.21716,1.21766,1.21709,1.21734,378,100.00000,78.63009,65.64637,45.58005,63.81245,33.55855,17.32140,0.65000,0.65000,0.65000,0.65000,0.65000,0.65000,0.65000,0.35000,0.35000,0.35000,0.35000,0.35000,0.35000,0.35000,0.41588,0.21763,0.31113,0.54411,0.08944,0.00000,0.02552,0.00000,0.00000,0.00000,0.00000,2.00000,2.00000,2.00000,0.40844,0.69315,0.55138,0.11688,0.02847,0.18657,0.75962,51.77986,47.12959,51.79845,50.29503,40.44482,47.69094,53.91591,53.76688,45.78000,50.92046,48.21420,40.44482,53.99928,53.91591,61.81059,45.78000,50.92046,48.21420,48.29269,72.92432,59.19549,57.78874,43.75562,48.88555,45.09294,42.59043,63.46180,54.09014,53.76688,41.73124,46.85065,41.97168,36.88817,53.99928,48.98480,51.77986,47.12959,51.79845,50.29503,40.44482,47.69094,53.91591,2.00000,1.00000,1.00000,1.00000,0.00000,2.00000,0.00000,1.21734,1.21734,1.21734,1.21703,1.21662,1.21614,1.21614,1.00000,0.00000,0.00000,0.00000,0.00000,0.00000,54.09014,1750.00000,1582.00000,1181.00000,2149.00000,2005.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,3968.00000,6478.00000,1100.60000,830.60000,710.40000,1528.80000,2338.00000,4059.80000,11957.40000,1100.60000,1028.40000,1549.00000,1999.20000,2805.00000,5243.40000,11403.60000,1100.60000,1112.20000,1828.00000,3417.20000,3211.40000,5719.60000,20836.20000,58.49142,54.86280,51.74028,41.78421,34.23924,31.59383,38.16118,0.00399,0.00440,0.00590,0.00383,0.00181,0.01609,0.02090,1.00000,1.00000,1.00000,1.00000,2.00000,2.00000,2.00000,0.00130,0.00115,0.00129,0.00165,0.00357,0.00770,0.00675,0.00400,0.00400,0.00400,0.00400,0.00400,0.00400,0.00400,-0.00002,-0.00005,-0.00022,-0.00057,-0.00090,-0.00093,-0.00012,-0.00008,-0.00009,-0.00028,-0.00072,-0.00098,-0.00054,0.00072,75.84098,66.06218,49.23077,44.68599,36.87375,26.81954,10.96674,48.69044,39.47233,34.94506,54.62237,45.50368,17.83720,11.65563,0.59748,0.50296,-0.04869,-0.59236,-0.82287,-0.80059,-0.43458,2.00000,2.00000,4.00000,6.00000,7.00000,7.00000,5.00000,378.00000,481.00000,741.00000,1308.00000,2005.00000,3968.00000,6478.00000,1.00000,1.00000,3.00000,1.00000,2.00000,2.00000,2.00000,0.05670,0.14430,0.82992,0.26160,1.34335,2.46016,17.87928"
#
# msg_parts=msg.split('|')
# columns=msg_parts[0]
# columns = columns.split(',')
# del columns[0]
#
# date_string=msg_parts[1].split(',')[0]
# features=np.array(msg_parts[1].split(',')[1:])
# features=[float(i) for i in features]
#
# data = np.array([features])
#
# df = pd.DataFrame(data, columns=columns)
# df=df.astype('float')
#
#
# best_model_path = os.path.join('.', 'best_models', 'EURUSD1')
#
# colums_needed = list(pd.read_csv(os.path.join(best_model_path, 'columns_needed.csv'), header=None).T.values[0])
# df = df[colums_needed]
# print(df.head())
# x_test = df.to_numpy().astype(float)
#
#
# from sklearn.impute import SimpleImputer
# my_imputer = SimpleImputer()
# x_test = my_imputer.fit_transform(x_test)
# from sklearn.preprocessing import MinMaxScaler
# import joblib
#
# # print(mm_scaler.data_max_)
# # print(mm_scaler.data_max_)
# # mm_scaler = MinMaxScaler(feature_range=(0, 1))  # or StandardScaler?
# mm_scaler = joblib.load(os.path.join(best_model_path, 'mm_scaler.joblib'))
# x_test = mm_scaler.transform(x_test)
# print(3, x_test)
# # print(mm_scaler.data_max_.shape)
# # print(mm_scaler.data_min_.shape)
# quit()
best_model_path = os.path.join('.', 'best_models', 'EURUSD1')

colums_needed = list(pd.read_csv(os.path.join(best_model_path, 'columns_needed.csv'), header=None).T.values[0])
msg="date,open,high,low,close,volume,mfi(5)_T3_B0,mfi(5)_T4_B0,mfi(5)_T6_B0,mfi(5)_T10_B0,mfi(5)_T15_B0,mfi(5)_T30_B0,mfi(5)_T60_B0,laguerre_rsi_wi_T3_B0,laguerre_rsi_wi_T4_B0,laguerre_rsi_wi_T6_B0,laguerre_rsi_wi_T10_B0,laguerre_rsi_wi_T15_B0,laguerre_rsi_wi_T30_B0,laguerre_rsi_wi_T60_B0,laguerre_rsi_wi_T3_B1,laguerre_rsi_wi_T4_B1,laguerre_rsi_wi_T6_B1,laguerre_rsi_wi_T10_B1,laguerre_rsi_wi_T15_B1,laguerre_rsi_wi_T30_B1,laguerre_rsi_wi_T60_B1,laguerre_rsi_wi_T3_B2,laguerre_rsi_wi_T4_B2,laguerre_rsi_wi_T6_B2,laguerre_rsi_wi_T10_B2,laguerre_rsi_wi_T15_B2,laguerre_rsi_wi_T30_B2,laguerre_rsi_wi_T60_B2,laguerre_rsi_wi_T3_B3,laguerre_rsi_wi_T4_B3,laguerre_rsi_wi_T6_B3,laguerre_rsi_wi_T10_B3,laguerre_rsi_wi_T15_B3,laguerre_rsi_wi_T30_B3,laguerre_rsi_wi_T60_B3,laguerre_rsi_wi_T3_B4,laguerre_rsi_wi_T4_B4,laguerre_rsi_wi_T6_B4,laguerre_rsi_wi_T10_B4,laguerre_rsi_wi_T15_B4,laguerre_rsi_wi_T30_B4,laguerre_rsi_wi_T60_B4,3_minutes_relat_T3_B0,3_minutes_relat_T4_B0,3_minutes_relat_T6_B0,3_minutes_relat_T10_B0,3_minutes_relat_T15_B0,3_minutes_relat_T30_B0,3_minutes_relat_T60_B0,3_minutes_relat_T3_B1,3_minutes_relat_T4_B1,3_minutes_relat_T6_B1,3_minutes_relat_T10_B1,3_minutes_relat_T15_B1,3_minutes_relat_T30_B1,3_minutes_relat_T60_B1,3_minutes_relat_T3_B2,3_minutes_relat_T4_B2,3_minutes_relat_T6_B2,3_minutes_relat_T10_B2,3_minutes_relat_T15_B2,3_minutes_relat_T30_B2,3_minutes_relat_T60_B2,3_minutes_relat_T3_B3,3_minutes_relat_T4_B3,3_minutes_relat_T6_B3,3_minutes_relat_T10_B3,3_minutes_relat_T15_B3,3_minutes_relat_T30_B3,3_minutes_relat_T60_B3,3_minutes_relat_T3_B4,3_minutes_relat_T4_B4,3_minutes_relat_T6_B4,3_minutes_relat_T10_B4,3_minutes_relat_T15_B4,3_minutes_relat_T30_B4,3_minutes_relat_T60_B4,3_minutes_relat_T3_B5,3_minutes_relat_T4_B5,3_minutes_relat_T6_B5,3_minutes_relat_T10_B5,3_minutes_relat_T15_B5,3_minutes_relat_T30_B5,3_minutes_relat_T60_B5,3_minutes_relat_T3_B6,3_minutes_relat_T4_B6,3_minutes_relat_T6_B6,3_minutes_relat_T10_B6,3_minutes_relat_T15_B6,3_minutes_relat_T30_B6,3_minutes_relat_T60_B6,3_minutes_relat_T3_B7,3_minutes_relat_T4_B7,3_minutes_relat_T6_B7,3_minutes_relat_T10_B7,3_minutes_relat_T15_B7,3_minutes_relat_T30_B7,3_minutes_relat_T60_B7,3_minutes_relat_T3_B8,3_minutes_relat_T4_B8,3_minutes_relat_T6_B8,3_minutes_relat_T10_B8,3_minutes_relat_T15_B8,3_minutes_relat_T30_B8,3_minutes_relat_T60_B8,fxc_activityfor_T3_B0,fxc_activityfor_T4_B0,fxc_activityfor_T6_B0,fxc_activityfor_T10_B0,fxc_activityfor_T15_B0,fxc_activityfor_T30_B0,fxc_activityfor_T60_B0,fxc_activityfor_T3_B1,fxc_activityfor_T4_B1,fxc_activityfor_T6_B1,fxc_activityfor_T10_B1,fxc_activityfor_T15_B1,fxc_activityfor_T30_B1,fxc_activityfor_T60_B1,fxc_activityfor_T3_B2,fxc_activityfor_T4_B2,fxc_activityfor_T6_B2,fxc_activityfor_T10_B2,fxc_activityfor_T15_B2,fxc_activityfor_T30_B2,fxc_activityfor_T60_B2,fxc_activityfor_T3_B3,fxc_activityfor_T4_B3,fxc_activityfor_T6_B3,fxc_activityfor_T10_B3,fxc_activityfor_T15_B3,fxc_activityfor_T30_B3,fxc_activityfor_T60_B3,fxc_activityfor_T3_B4,fxc_activityfor_T4_B4,fxc_activityfor_T6_B4,fxc_activityfor_T10_B4,fxc_activityfor_T15_B4,fxc_activityfor_T30_B4,fxc_activityfor_T60_B4,fxc_activityfor_T3_B5,fxc_activityfor_T4_B5,fxc_activityfor_T6_B5,fxc_activityfor_T10_B5,fxc_activityfor_T15_B5,fxc_activityfor_T30_B5,fxc_activityfor_T60_B5,rsi(14)_T3_B0,rsi(14)_T4_B0,rsi(14)_T6_B0,rsi(14)_T10_B0,rsi(14)_T15_B0,rsi(14)_T30_B0,rsi(14)_T60_B0,macd(12_26_9)_T3_B0,macd(12_26_9)_T4_B0,macd(12_26_9)_T6_B0,macd(12_26_9)_T10_B0,macd(12_26_9)_T15_B0,macd(12_26_9)_T30_B0,macd(12_26_9)_T60_B0,macd(12_26_9)_T3_B1,macd(12_26_9)_T4_B1,macd(12_26_9)_T6_B1,macd(12_26_9)_T10_B1,macd(12_26_9)_T15_B1,macd(12_26_9)_T30_B1,macd(12_26_9)_T60_B1,stoch(8_5_3)_T3_B0,stoch(8_5_3)_T4_B0,stoch(8_5_3)_T6_B0,stoch(8_5_3)_T10_B0,stoch(8_5_3)_T15_B0,stoch(8_5_3)_T30_B0,stoch(8_5_3)_T60_B0,stoch(8_5_3)_T3_B1,stoch(8_5_3)_T4_B1,stoch(8_5_3)_T6_B1,stoch(8_5_3)_T10_B1,stoch(8_5_3)_T15_B1,stoch(8_5_3)_T30_B1,stoch(8_5_3)_T60_B1,murreys_math_os_T3_B0,murreys_math_os_T4_B0,murreys_math_os_T6_B0,murreys_math_os_T10_B0,murreys_math_os_T15_B0,murreys_math_os_T30_B0,murreys_math_os_T60_B0,murreys_math_os_T3_B1,murreys_math_os_T4_B1,murreys_math_os_T6_B1,murreys_math_os_T10_B1,murreys_math_os_T15_B1,murreys_math_os_T30_B1,murreys_math_os_T60_B1,3_minutes_relat_T3_B0,3_minutes_relat_T4_B0,3_minutes_relat_T6_B0,3_minutes_relat_T10_B0,3_minutes_relat_T15_B0,3_minutes_relat_T30_B0,3_minutes_relat_T60_B0,3_minutes_relat_T3_B1,3_minutes_relat_T4_B1,3_minutes_relat_T6_B1,3_minutes_relat_T10_B1,3_minutes_relat_T15_B1,3_minutes_relat_T30_B1,3_minutes_relat_T60_B1,3_minutes_relat_T3_B2,3_minutes_relat_T4_B2,3_minutes_relat_T6_B2,3_minutes_relat_T10_B2,3_minutes_relat_T15_B2,3_minutes_relat_T30_B2,3_minutes_relat_T60_B2,3_minutes_relat_T3_B3,3_minutes_relat_T4_B3,3_minutes_relat_T6_B3,3_minutes_relat_T10_B3,3_minutes_relat_T15_B3,3_minutes_relat_T30_B3,3_minutes_relat_T60_B3,3_minutes_relat_T3_B4,3_minutes_relat_T4_B4,3_minutes_relat_T6_B4,3_minutes_relat_T10_B4,3_minutes_relat_T15_B4,3_minutes_relat_T30_B4,3_minutes_relat_T60_B4,3_minutes_relat_T3_B5,3_minutes_relat_T4_B5,3_minutes_relat_T6_B5,3_minutes_relat_T10_B5,3_minutes_relat_T15_B5,3_minutes_relat_T30_B5,3_minutes_relat_T60_B5,3_minutes_relat_T3_B6,3_minutes_relat_T4_B6,3_minutes_relat_T6_B6,3_minutes_relat_T10_B6,3_minutes_relat_T15_B6,3_minutes_relat_T30_B6,3_minutes_relat_T60_B6,3_minutes_relat_T3_B7,3_minutes_relat_T4_B7,3_minutes_relat_T6_B7,3_minutes_relat_T10_B7,3_minutes_relat_T15_B7,3_minutes_relat_T30_B7,3_minutes_relat_T60_B7,3_minutes_relat_T3_B8,3_minutes_relat_T4_B8,3_minutes_relat_T6_B8,3_minutes_relat_T10_B8,3_minutes_relat_T15_B8,3_minutes_relat_T30_B8,3_minutes_relat_T60_B8|2021.01.15 22:21,1.20801,1.20812,1.20793,1.20800,88,72.49933,56.46207,61.09380,33.37266,52.37033,16.39772,34.39006,0.65000,0.65000,0.65000,0.65000,0.65000,0.65000,0.65000,0.35000,0.35000,0.35000,0.35000,0.35000,0.35000,0.35000,0.38497,0.15958,0.07880,0.14745,0.01530,0.00000,0.00000,0.00000,0.00000,2.00000,2.00000,2.00000,2.00000,2.00000,0.15769,0.34902,0.34777,0.07362,0.07738,0.03890,0.03525,44.20517,45.47082,55.14943,51.98386,36.51342,48.84477,29.92817,44.20517,45.47082,53.22621,50.45858,37.36093,48.45283,33.45803,44.98007,47.06196,53.22621,50.45858,43.61689,48.45283,45.01558,43.27874,44.54524,50.34137,48.17066,40.48891,46.81029,39.23681,41.57741,42.02853,47.45653,45.88273,37.36093,45.16776,33.45803,44.20517,45.47082,55.14943,51.98386,36.51342,48.84477,29.92817,0.00000,0.00000,1.00000,1.00000,2.00000,1.00000,2.00000,1.20800,1.20800,1.20800,1.20794,1.20794,1.20794,1.20784,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,181.00000,130.00000,259.00000,363.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,1974.00000,7888.00000,88.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,68.40000,234.80000,284.40000,764.00000,934.00000,3385.00000,8772.20000,294.20000,382.80000,618.20000,1189.20000,2627.60000,5655.60000,9201.80000,636.80000,741.80000,1414.20000,1940.20000,4631.40000,5715.00000,9338.80000,45.85716,44.54752,40.31201,38.13930,33.55128,30.38605,24.49676,-0.00009,-0.00015,-0.00027,-0.00052,-0.00087,-0.00148,-0.00182,-0.00012,-0.00017,-0.00029,-0.00060,-0.00093,-0.00138,-0.00131,76.25899,57.54190,45.77778,41.66667,15.94595,12.12553,2.90023,59.01056,41.74470,28.32831,36.16202,23.24932,14.04085,13.23907,-0.16667,-0.39655,-0.70464,-0.83090,-0.89735,-0.92526,-0.96285,4.00000,5.00000,6.00000,7.00000,7.00000,7.00000,7.00000,44.20517,45.47082,55.14943,51.98386,36.51342,48.84477,29.92817,44.20517,45.47082,53.22621,50.45858,37.36093,48.45283,33.45803,44.98007,47.06196,53.22621,50.45858,43.61689,48.45283,45.01558,43.27874,44.54524,50.34137,48.17066,40.48891,46.81029,39.23681,41.57741,42.02853,47.45653,45.88273,37.36093,45.16776,33.45803,44.20517,45.47082,55.14943,51.98386,36.51342,48.84477,29.92817,0.00000,0.00000,1.00000,1.00000,2.00000,1.00000,2.00000,1.20800,1.20800,1.20800,1.20794,1.20794,1.20794,1.20784,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000"
msg_parts = msg.split('|')
columns = msg_parts[0]
columns = columns.split(',')
del columns[0]
date_string = msg_parts[1].split(',')[0]
features = np.array(msg_parts[1].split(',')[1:])
data = np.array([features]).astype(np.float)
df = pd.DataFrame(data, columns=columns)

df = df['3_minutes_relat_T3_B0.1']


# x_test = df.to_numpy()
# from sklearn.impute import SimpleImputer
# my_imputer = SimpleImputer()
# x_test = my_imputer.fit_transform(x_test)
#
# df_testing = pd.read_csv(metatrader_dir + "Testing.csv")
# df_testing = df_testing[colums_needed]
# x_test_testing = df_testing.to_numpy()
# x_test_testing = my_imputer.fit_transform(x_test_testing)
#
# from sklearn.preprocessing import MinMaxScaler
# # mm_scaler = MinMaxScaler(feature_range=(0, 1))  # or StandardScaler?
# import joblib
# mm_scaler = joblib.load(os.path.join(best_model_path, 'mm_scaler.joblib'))
# print(mm_scaler.data_min_)
# # mm_scaler = mm_scaler.fit(x_test_testing)
# #
# x_test = mm_scaler.transform(x_test)
# print(x_test)

# import joblib
# mm_scaler = joblib.load(os.path.join(best_model_path, 'mm_scaler.joblib'))
# print(mm_scaler.data_min_,)
# x_test = mm_scaler.fit(x_test)
# print(x_test)

