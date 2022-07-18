# 라이브러리 import
# 기본적인 라이브러리 import
import xlwings as xw
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time
from tqdm import tqdm
# from sqlalchemy import create_engine
# import pymysql
import filetransfer as ft

# import win32com.client

# import ctypes
# excel = ("C:\\Program Files (x86)\\Microsoft Office\\Office15\\excel.exe")

# ctypes.windll.shell32.ShellExecuteA(0, 'open', excel, None, None, 1)

# excel = win32com.client.Dispatch("Excel.Application")
# excel.Visible = True

'''
채권목록 구하기
'''

# 채권 발행일과 만기일 기준으로 종목정보를 조회하는 bondlist 엑셀파일에서 이표채의 KR코드와 종목명을 가져옵니다.
print("loading bondlist.xlsx")
sheet_bondlist = xw.Book(r'bondlist.xlsx').sheets[0]
startcell = 'A8'
maxrow = sheet_bondlist.range(startcell).end('down').row
maxcol = sheet_bondlist.range(startcell).end('right').column
df = sheet_bondlist.range(startcell,(maxrow,maxcol)).options(pd.DataFrame, index=False).value
df_coupon = df[df['이자종류'] == "이표채"]
codes = df_coupon['표준코드'].tolist()
setcodes = set(codes)
names = [name[13:-1] for name in df_coupon['종목명'].tolist()]
print(f"total ktb bond count : {len(names)}")



'''
채권수급데이터 가져오기
'''
# 채권데이터 엑셀의 시트들을 가져오기
print("loading bonddata.xlsx")
sheets_bonddata = xw.Book(r'bonddata.xlsx').sheets
sheetnames = set() # bondlist의 채권 목록과 비교를 위해 시트의 이름(KR코드) 집합생성
for sheet in tqdm(sheets_bonddata):
    sheetnames.add(sheet.name)

codesToAdd = setcodes - sheetnames # 엑셀에 없는 추가할 채권목록을 구함
print(codesToAdd)
if len(codesToAdd) != 0:
    print("[채권데이터 엑셀] 추가해야할 종목이 있습니다.")
    print("추가할 종목 :" + str(codesToAdd))
    strInput = input("자동추가 시도하시려면 y키를 눌러주세요.")
    if (strInput == 'y' or strInput =='Y') :
        sheets_bonddata[-1].range('A1:AH9').copy()
        for code in tqdm(codesToAdd):
            # 시트 자체를 복붙하는 버전
            newsheet = sheets_bonddata[-1].copy()# 시트를 하나 복사하고
            newsheet.name = code

            # sheets_bonddata[-1].copy(name=code)# 시트를 하나 복사하고
            time.sleep(1)
            sheets_bonddata[code].range('B3').value = code # 종목코드 셀에 해당 종목코드를 투입

            
            # # sheet를 add한후 내용 복붙하는 버전
            # sheets_bonddata.add(name=code, after=sheets_bonddata[-1])
            # # sheets_bonddata.add(name=code)
            # time.sleep(2)
            # sheets_bonddata[code].range('A1').paste()
            # time.sleep(2)
            # sheets_bonddata[code].range('B3').value = code
            # time.sleep(2)
            # sheets_bonddata[code].autofit()
    time.sleep(2)

# 조회가 끝난 각 시트를 돌면서 데이터를 받아서 하나의 DataFrame으로 만든다.
keycell = 'A9'
df = pd.DataFrame()
for sheet in tqdm(sheets_bonddata):
    maxrow = sheet.range(keycell).end('down').row
    maxcol = sheet.range(keycell).end('right').column
    bondname = sheet.range('B4').value[13:-1]
    tempdf = sheet.range(keycell,(maxrow,maxcol)).options(pd.DataFrame, index=False).value
    tempdf['종목명'] = bondname
    tempdf.set_index(['일자','종목명'], inplace=True)
    tempdf_notnull = tempdf[tempdf['민평3사 수익률(산출일) 당일'].notnull()]
    df = df.append(tempdf_notnull)

# DataFrame의 컬럼명을 적당한 이름들로 수정
df.columns = [
    '민평금리', '민평금리전일비', '민평가격', '듀레이션', 
    '전체순매수', '외국인순매수', '은행순매수', '보험기금순매수', '자산운용공모순매수', '자산운용사모순매수', '종금순매수', '정부순매수', '기타법인순매수', '개인순매수', 
    '장내거래량', '장외거래량',
    '대차거래', '대차상환', '대차잔량', 
    '증권대여', '보험대여', '투신대여', '은행대여', '연기금대여', '외국인대여', '기타대여', 
    '증권차입', '보험차입', '투신차입', '은행차입', '연기금차입', '외국인차입', '기타차입'
    ]

# 컬럼 추가 및 수정
# 대차증감 = 거래 - 상환
# 대차 전일잔량 삭제
df['대차증감'] = df['대차거래'] - df['대차상환']

# NA처리 (0으로 변경) - 문제소지는 민평금리 전일비가 일부(종목의 상장일) 0이 되는문제? ->딱히 문제는 없을듯 
df.fillna(0, inplace=True)
# df.isna().sum()

# 각 컬럼의 type을 적절하게 수정해준다.
df[['전체순매수', '외국인순매수', '은행순매수', '보험기금순매수', '자산운용공모순매수', '자산운용사모순매수', 
    '종금순매수', '정부순매수', '기타법인순매수', '개인순매수', '장내거래량', '장외거래량', 
    '대차거래', '대차상환', '대차잔량', 
    '증권대여', '보험대여', '투신대여', '은행대여', '연기금대여', '외국인대여', '기타대여', 
    '증권차입', '보험차입', '투신차입', '은행차입', '연기금차입', '외국인차입', '기타차입', '대차증감']] = df[['전체순매수', '외국인순매수', '은행순매수', '보험기금순매수', '자산운용공모순매수', '자산운용사모순매수', 
    '종금순매수', '정부순매수', '기타법인순매수', '개인순매수', '장내거래량', '장외거래량', 
    '대차거래', '대차상환', '대차잔량', 
    '증권대여', '보험대여', '투신대여', '은행대여', '연기금대여', '외국인대여', '기타대여', 
    '증권차입', '보험차입', '투신차입', '은행차입', '연기금차입', '외국인차입', '기타차입', '대차증감']].astype(np.int64)

# 만든 DataFrame을 pickle 로 떨궈본다.
df.to_pickle('pkl_bonddata.pkl')
print("pkl_bonddata.pkl saved.")


'''
채권발행량 데이터 가져오기
'''
# 채권발행량 엑셀의 시트들을 가져오기
print("loading bondbalance.xlsx")
sheets_bondbalance = xw.Book(r'bondbalance.xlsx').sheets
sheetnames = set() # bondlist의 채권 목록과 비교를 위해 시트의 이름(KR코드) 집합생성
for sheet in tqdm(sheets_bondbalance):
    sheetnames.add(sheet.name)
    
codesToAdd = setcodes - sheetnames # 엑셀에 없는 추가할 채권목록을 구함
print(codesToAdd)
if len(codesToAdd) != 0:
    print("[채권발행량 엑셀]추가해야할 종목이 있습니다.")
    print("추가할 종목 :" + str(codesToAdd))
    strInput = input("자동추가 시도하시려면 y키를 눌러주세요.")
    if (strInput == 'y' or strInput =='Y') :
        sheets_bondbalance[-1].range('A1:AH9').copy()
        for code in tqdm(codesToAdd):
            # 시트 자체를 복붙하는 버전
            newsheet = sheets_bondbalance[-1].copy()# 시트를 하나 복사하고
            newsheet.name = code

            # sheets_bonddata[-1].copy(name=code)# 시트를 하나 복사하고
            time.sleep(1)
            sheets_bondbalance[code].range('B3').value = code # 종목코드 셀에 해당 종목코드를 투입

            
            # # sheet를 add한후 내용 복붙하는 버전
            # sheets_bonddata.add(name=code, after=sheets_bonddata[-1])
            # # sheets_bonddata.add(name=code)
            # time.sleep(2)
            # sheets_bonddata[code].range('A1').paste()
            # time.sleep(2)
            # sheets_bonddata[code].range('B3').value = code
            # time.sleep(2)
            # sheets_bonddata[code].autofit()
    time.sleep(2)

# 조회가 끝난 각 시트를 돌면서 데이터를 받아서 하나의 DataFrame으로 만든다.
keycell = 'A9'
df = pd.DataFrame()
for sheet in tqdm(sheets_bondbalance):
    if sheet.range('A10').value != None: # 상장잔액증감이 없는 종목들 처리
        maxrow = sheet.range(keycell).end('down').row
        maxcol = sheet.range(keycell).end('right').column
        print(sheet.range('B4').value)
        bondname = sheet.range('B4').value[13:-1]
        tempdf = sheet.range(keycell,(maxrow,maxcol)).options(pd.DataFrame, index=False).value
        tempdf['종목명'] = bondname
        tempdf.set_index(['일자','종목명'], inplace=True)
        # tempdf_notnull = tempdf[tempdf['민평3사 수익률(산출일) 당일'].notnull()]
        df = df.append(tempdf)

# 컬럼명은 문제가 없으므로 컬럼 타입만 int64로 변경
df[['상장잔액증감']] = df[['상장잔액증감']].astype(np.int64)
# df[['상장잔액증감']] = df[['상장잔액증감']].fillna(0).astype(np.int64) # pandas.errors.IntCastingNaNError: Cannot convert non-finite values (NA or inf) to integer

# 만든 DataFrame을 pickle 로 떨궈본다.
df.to_pickle('pkl_bondbalance.pkl')
print("pkl_bondbalance.pkl saved.")

'''''''''''''''''''''
시장금리 DB저장
'''''''''''''''''''''
sheets_marketdata = xw.Book(r'db_marketdata.xlsx').sheets[0]
time.sleep(2)
keycell = 'A4'
df = pd.DataFrame()
if sheet.range('A5').value != None: # 조회데이터 없는경우 처리
    maxrow = sheets_marketdata.range(keycell).end('down').row
    maxcol = sheets_marketdata.range(keycell).end('right').column
    tempdf = sheets_marketdata.range(keycell,(maxrow,maxcol)).options(pd.DataFrame, index=False).value
    tempdf.set_index('일자', inplace=True)
    # tempdf_notnull = tempdf[tempdf['민평3사 수익률(산출일) 당일'].notnull()]
    df = df.append(tempdf)

# DataFrame의 컬럼명을 적당한 이름들로 수정
df.columns = ['국고3년', '국고5년', '국고10년', '국고20년', '국고30년',
       '국고3년전일대비', '국고5년전일대비', '국고10년전일대비', '국고20년전일대비',
       '국고30년전일대비']
# 만든 DataFrame을 pickle 로 떨궈본다.
df.to_pickle('pkl_marketdata.pkl')
print("pkl_marketdata.pkl saved.")
'''''''''''''''''''''
입찰캘린더 DB저장
'''''''''''''''''''''
# 엑셀파일에서 DataFrame 만들기. 종목명 작업.
sheet_calendar = xw.Book(r'db_calendar.xlsx').sheets[0]
keycell = 'A2'
maxrow = sheet_calendar.range(keycell).end('down').row
maxcol = sheet_calendar.range(keycell).end('right').column
df = sheet_calendar.range(keycell,(maxrow,maxcol)).options(pd.DataFrame, index=False).value
df['종목명'] = df['전체종목명'].str[13:-1]

#발행년월 에서 발행년도, 발행월을 뽑아내는 다소 tricky한 부분
df['발행년도'] = df['발행년월'].astype(int)
df['발행월'] = np.round((df['발행년월']%1*100)).astype(int)
df.drop(['발행년월'], axis=1, inplace=True)
df.set_index(['일자','종목명'], inplace=True)
# NA처리 (0으로 변경) - 공란인 것들이 좀 있음
df.fillna(0, inplace=True)
# 만든 DataFrame을 pickle 로 떨궈본다.
df.to_pickle('pkl_calendar.pkl')
print("pkl_calendar.pkl saved.")


# 파일 업로드 부분
ssh_manager = ft.SSHManager()
# ip = "hanatrading.ml"
ip = "211.216.15.238"
ssh_manager.create_ssh_client(ip, 2222, "dql", "Cnts13!")
datapath = "~/projects/AH/assets/"
ssh_manager.send_file("pkl_bonddata.pkl", datapath)
print("pkl_bonddata.pkl uploaded.")
ssh_manager.send_file("pkl_bondbalance.pkl", datapath)
print("pkl_bondbalance.pkl uploaded.")
ssh_manager.send_file("pkl_marketdata.pkl", datapath)
print("pkl_marketdata.pkl uploaded.")
ssh_manager.send_file("pkl_calendar.pkl", datapath)
print("pkl_calendar.pkl uploaded.")
ssh_manager.close_ssh_client()
# db_connection_str = 'mysql+pymysql'

# for code, name in zip(codes, names):
#     newsheet = sheets[0].copy(name=code) # 이표채 종목수만큼 시트를 복제, 시트명은 KR코드

