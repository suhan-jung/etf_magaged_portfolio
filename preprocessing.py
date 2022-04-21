#!/usr/bin/env python
# coding: utf-8

# # 데이터 전처리

# 필요 라이브러리 로드 및 환경변수 설정

# In[1]:


# get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import referenceBL
import datetime
import importlib
importlib.reload(referenceBL)

plt.style.use('seaborn-whitegrid')
plt.rc('font', family="Malgun Gothic")
plt.rc('axes', unicode_minus=False)


# ### Bloomberg 가격 데이터 불러오기

# In[3]:


# 데이터 로딩하려는 대상 티커 리스트를 가져온다.
# file_tickers = 'tickers.csv'
file_tickers = 'snp_top50.csv'
tickers = pd.read_csv(file_tickers, header=None)
tickers = tickers[0].tolist()
tickers


# In[4]:


# 각 티커별로 이름, 통화, 거래소, 종목유형, 생성일, 수정일을 저장할 데이터프레임을 생성한다.
blp = referenceBL.BLPInterface()
master_table = []
for ticker in tickers:
    temp = []
    try:
        currency = blp.referenceRequest(securities=ticker, fields='CRNCY')
        temp.append(currency)
        exchange = blp.referenceRequest(securities=ticker, fields='CDR_EXCH_CODE')
        temp.append(exchange)
        description = blp.referenceRequest(securities=ticker, fields='name')
        temp.append(description)
        sec_type = blp.referenceRequest(securities=ticker, fields='SECURITY_TYP')
        temp.append(sec_type)
        created_date = datetime.datetime.now()
        temp.append(created_date)
        last_updated_date = datetime.datetime.now()
        temp.append(last_updated_date)
        master_table.append(temp)
    except Exception:
        print("{} was not completed for master table".format(ticker))
        pass
blp.close()
master_df = pd.DataFrame(master_table, index=tickers,
                            columns=['CURRENCY', 'EXC_CODE', 'DESCRIPTION',
                                    'TYPE', 'CREATED_DATE', 'UPDATED_DATE'])
master_df['BLCODE'] = tickers
master_df.index.names = ['EQID']
save_id_df = 'master_df.csv'
master_df.to_csv(save_id_df)


# In[5]:


# 실제로 historicalRequest를 통해 데이터를 가져온다.
blp = referenceBL.BLPInterface()
today = datetime.datetime.today()
today = '{:02d}{:02d}{:02d}'.format(today.year, today.month, today.day)
data_table_all = []
for ticker in tickers:
    data_name = blp.referenceRequest(ticker, 'NAME')
    data_table = blp.historicalRequest(
        securities=ticker, 
        fields=['PX_Last'], 
        startDate='20000103', 
        endDate=today)
    data_table.columns = [ticker]
    data_table_all.append(data_table)
    print(ticker)

blp.close()
data = pd.concat(data_table_all, axis=1)

save_id_data = 'data.csv'
data.to_csv(save_id_data)


# ### 데이터 전처리(N/A 제거)

# In[10]:


data = data.dropna(axis=1)


# In[9]:


# ./data 폴더의 BloombergMain.py(융기원 코드)를 통해 생성한 csv파일을 불러온다.
data = pd.read_csv('data.csv', index_col=0, parse_dates=True)


# In[11]:


def preprocessing(data):
    # 일자별 종가인 dataframe을 받아서 nan값 제거, linear interpolation을 한후 로그일간수익율로 변환하여 반환한다.
    df = data.dropna(thresh=4)
    df = df.interpolate(method='linear', limit_direction='forward')  # 연휴에 따른 급격한 변화를 smoothing해주기 위해 interpolation
    df = df.dropna()
    dr = np.log(df).diff(1).dropna()
    return dr


# In[12]:


df = preprocessing(data)
df.to_csv('data_preprocessed.csv')


# In[ ]:




