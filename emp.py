import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import tensorflow as tf
from data import referenceBL
import functions as fns


#폴더 내에 전처리 완료파일, 원파일 존재여부
file_pp_exist = False
file_raw_exist = False

date_today = datetime.today().strftime('%Y%m%d')

filename_pp = f'data_pp_{date_today}'
filename_raw = f'data_raw_{date_today}'

print("today's date is",date_today)
files = filter(os.path.isfile, os.listdir(os.curdir))
for file in files:
    if (filename_pp) in file:
        print("preprocessed file exists")
        file_pp_exist = True
    if (filename_raw) in file:
        print("raw file exists.")
        file_raw_exist = True
    if file_pp_exist & file_raw_exist:
        break

#전처리된 파일이 없는데
if not file_pp_exist:
    print("preprocessed file doesn't exist")
    #raw 파일도 없으면 
    if not file_raw_exist:
        print("raw file doesn't exist")
        #raw 파일을 만든다.
        tickers = pd.read_csv("tickers.csv", header=None)
        tickers = tickers[0].tolist()

        blp = referenceBL.BLPInterface()
        data_table_all = []
        for ticker in tqdm(tickers):
            data_name = blp.referenceRequest(ticker, 'NAME')
            data_table = blp.historicalRequest(
                securities=ticker, 
                fields=['PX_Last'], 
                startDate='20000103', 
                endDate=date_today)
            data_table.columns = [ticker]
            data_table_all.append(data_table)
            print(ticker)

        blp.close()
        data = pd.concat(data_table_all, axis=1)
        data.to_csv(filename_raw)

    #raw 파일은 있으면 전처리만 한다.
    data = pd.read_csv(filename_raw, index_col=0, parse_dates=True)
    df = fns.preprocessing(data)
    df.to_csv(filename_pp)

#filename_pp를 가지고 시작
# 로그수익율로 전처리 완료된 파일을 불러온다.
dr = pd.read_csv(filename_pp, index_col=0, parse_dates=True)

# train / test set 준비 - train data : test data = 80 : 20
train_index = int(dr.shape[0]*0.8)
print(train_index)
# 80:20 으로 train / test 분할 (numpy array로 변환)
train_data = dr[:train_index].values
train_date = dr[:train_index].index.values.astype('datetime64[D]')

test_data = dr[train_index:].values
test_date = dr[train_index:].index.values.astype('datetime64[D]')

# 자산배분비율 산출 기준 : 과거 60일 데이터로 산출, 미래 20일의 최적수익율로 최적화
window_size_past=30
window_size_future=10
xc_train, xf_train = fns.make_data_window(train_data, window_size_past, window_size_future)
xc_test, xf_test = fns.make_data_window(test_data, window_size_past, window_size_future)

norm_l = tf.keras.layers.Normalization()
norm_l.adapt(xc_train)
xc_train_norm = norm_l(xc_train)
xf_train_norm = norm_l(xf_train)
xc_test_norm = norm_l(xc_test)
xf_test_norm = norm_l(xf_test)

model, history = fns.model_build_fit(xc_train_norm, xf_train_norm, xc_test_norm, xf_test_norm, 250)

# loss trajectory를 확인한다.
plt.figure(figsize=(12,8))
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.legend()
plt.show()