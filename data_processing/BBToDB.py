import referenceBL
import pandas as pd
from tqdm import tqdm
import datetime
import filetransfer as ft

filename_tickers = "tickers.csv"
tickers = pd.read_csv(filename_tickers, header=None)
tickers = tickers[0].tolist()

# 각 티커별로 이름, 통화, 거래소, 종목유형, 생성일, 수정일을 저장할 데이터프레임을 생성한다.
blp = referenceBL.BLPInterface()
master_table = []
for ticker in tqdm(tickers):
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

# 실제로 historicalRequest를 통해 데이터를 가져온다.
blp = referenceBL.BLPInterface()
today = datetime.datetime.today()
today = '{:02d}{:02d}{:02d}'.format(today.year, today.month, today.day)
data_table_all = []
for ticker in tqdm(tickers):
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


# # 파일 업로드 부분
# ssh_manager = ft.SSHManager()
# # ip = "hanatrading.ml"
# ip = "211.216.15.238"
# ssh_manager.create_ssh_client(ip, 2222, "dql", "Cnts13!")
# datapath = "~/projects/portfolio_optimization/"
# ssh_manager.send_file("master_df.csv", datapath)
# print("master_df.csv uploaded.")
# ssh_manager.send_file("data.csv", datapath)
# print("data.csv uploaded.")