import pandas as pd
import datetime
import referenceBL

file_tickers = 'Tickers.csv'


def create_master_table(tickers_csv):
    tickers = pd.read_csv(tickers_csv, header=None)
    tickers = tickers[0].tolist()

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

    master_df = pd.DataFrame(master_table, index=tickers,
                             columns=['CURRENCY', 'EXC_CODE', 'DESCRIPTION',
                                      'TYPE', 'CREATED_DATE', 'UPDATED_DATE'])
    master_df['BLCODE'] = tickers
    master_df.index.names = ['EQID']

    return master_df


def getEquityData(tickers_csv, start_date='20000101'):
    blp = referenceBL.BLPInterface()

    tickers = pd.read_csv(tickers_csv, header=None)
    tickers = tickers[0].tolist()

    today = datetime.datetime.today()
    today = '{:02d}{:02d}{:02d}'.format(today.year, today.month, today.day)

    data_table_all = []
    for ticker in tickers:
        print(ticker)
        data_table = blp.historicalRequest(securities=ticker,
                                           fields=['PX_Last'], startDate=start_date, endDate=today)
        data_table.columns = [ticker]
        data_table_all.append(data_table)

    blp.close()
    data_table_all = pd.concat(data_table_all, axis=1)

    return data_table_all


if __name__ == "__main__":
    # if updateDB:
    #     updateEquityData()
    # else:
    today = datetime.datetime.today()

    master_df = create_master_table(file_tickers)
    # save_id_df = ('_').join(['sj_master_df', '{:02d}{:02d}{:02d}'.format(today.year, today.month,today.day), '.csv']) 
    save_id_df = 'master_df.csv'
    master_df.to_csv(save_id_df)

    data = getEquityData(file_tickers, '20110103')
    # save_id_data = ('_').join(['sj_data', '{:02d}{:02d}{:02d}'.format(today.year, today.month,today.day), '.csv'])
    save_id_data = 'data.csv'
    data.to_csv(save_id_data)