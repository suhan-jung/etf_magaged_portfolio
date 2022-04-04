import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# plot style settings
plt.style.use('seaborn-whitegrid')
plt.rc('font', family="Malgun Gothic")
plt.rc('axes', unicode_minus=False)

# 사전작업: ./data 폴더의 BloombergMain.py (융기원 코드) 활용하여 .csv 형태로 준비
data = pd.read_csv('./data/data.csv', index_col=0, parse_dates=True)
df = data.dropna(thresh=4)
# df = pd.read_csv('https://raw.githubusercontent.com/suhan-jung/portfolio_optimization/master/data/data.csv', index_col=0, parse_dates=True) # 구글 코랩등 사용을 위해 외부 링크로 준비
df = df.interpolate(method='linear', limit_direction='forward') # 연휴에 따른 급격한 변화를 smoothing해주는 것이 필요함
# df_dropna = df.dropna() # NA는 행 자체를 삭제(대세에 지장 없음) 하지만 장기적으로는 휴일 관련 interpolation을 통해 연휴에 따른 급격한 변화를 smoothing해주는 것이 필요함
print(df.shape)
df = df.dropna()
print(df.shape)
# df.columns = ["국내주식","선진국주식","신흥국주식","국내채권","선진국채권","신흥국채권","해외크레딧","부동산","원자재","금"]
print(df.head(5))
print(df.tail(5))

df_w = df.resample('W', closed='right').last()
df_w.head(10)

# simple return
dR = df.pct_change().dropna()
wR = df_w.pct_change().dropna()

# log return
dr = np.log(df).diff(1).dropna()
wr = np.log(df_w).diff(1).dropna()

# cumulative daily return
cdr = dr.cumsum()
cdR = (1+dR).cumprod()-1
# total sum of daily return
tdr = dr.sum() # total
tdR = (1+dR).prod()-1
# annualized daily return
adr = tdr/len(dr)*250
adR = (tdR+1)**(250/len(dr))-1

# standard deviation of daily return
sd_dr = dr.std()
sd_dR = dR.std()
# annualized standard deviation of daily return
asd_dr = sd_dr * np.sqrt(250)
asd_dR = sd_dR * np.sqrt(250)
# covariance of weekly return
cov_wr = wr.cov()
cov_wR = wR.cov()
# annualized covariance of weekly return
acov_wr = cov_wr * 52
acov_wR = cov_wR * 52
# correlation of weekly return
corr_wr = wr.corr()
corr_wR = wR.corr()

# yearly correlation of weekly return
ycorr_wr = wr.groupby(pd.Grouper(freq='Y')).corr()
ycorr_wR = wR.groupby(pd.Grouper(freq='Y')).corr()

# yearly return
yr = dr.groupby(pd.Grouper(freq='Y')).sum()
yR = (dR+1).groupby(pd.Grouper(freq='Y')).prod()-1
# yearly standard deviation of daily return
ysd_dr = dr.groupby(pd.Grouper(freq='Y')).std()*np.sqrt(250)
ysd_dR = dR.groupby(pd.Grouper(freq='Y')).std()*np.sqrt(250)

# j = 6
col = 3
fig, ax = plt.subplots(math.ceil(dr.shape[1]/col),col, figsize=(25,30), sharey=True, sharex=True)
# fig, ax = plt.subplots(6,2, figsize=(15,30))
for j in range(yr.shape[1]):
    x = j//col
    y= j%col
    ax[x][y].plot(ysd_dr.iloc[:,j], yr.iloc[:,j], marker="x", alpha=1.0)
    # ax[x][y].plot(ysdr.iloc[:,j], yr.iloc[:,j]-yearly_rf.iloc[:,0], marker=j, alpha=1.0)
    ax[x][y].set_title(yr.columns[j])
    ax[x][y].set_xlabel('Volatility')
    ax[x][y].set_ylabel('Return')
    ax[x][y].tick_params(axis='both', which='both', labelsize=12, labelbottom=True, labelleft=True)
    for i, label in enumerate(yr.index):
        ax[x][y].text(ysd_dr.iloc[:,j][i], yr.iloc[:,j][i], f"{label.strftime('%Y')}")
        # ax[x][y].text(ysdr.iloc[:,j][i], yr.iloc[:,j][i]-yearly_rf.iloc[:,0][i], f"{label.strftime('%Y')}")
        # plt.text(ysdr.iloc[:,j][i], yr.iloc[:,j][i], f"{yr.iloc[:,j].name}-{label.strftime('%Y')}")
# plt.text(x.iloc[:,0],y.iloc[:,0], 'text', fontsize=10)
# plt.xlim(left=0)
# plt.ylim(bottom=-0.4)
# plt.xlabel('Volatility (Std. Deviation)')
# plt.ylabel('Expected Returns')
# plt.title('Risk-Return Profiles of Various Assets')

# 전체자산 추가 plotting
# ax[5][1].scatter(asdr, adr, marker="x", s=100, alpha=1.0)
# for i, label in enumerate(adr.index):
#     ax[5][1].text(asdr[i]+0.002, adr[i], label)
# ax[5][1].set_title("전체 자산")
# ax[5][1].set_xlabel('Volatility')
# ax[5][1].set_ylabel('Return')
# ax[5][1].tick_params(axis='both', which='both', labelsize=12, labelbottom=True, labelleft=True)
plt.show()

plt.figure(figsize=(10,10))
plt.scatter(asd_dr, adr, marker="x", s=100, alpha=1.0)
for i, label in enumerate(adr.index):
    plt.text(asd_dr[i]+0.002, adr[i], label)
plt.xlim(left=-0.005)
plt.ylim(bottom=-0.05)
plt.xlabel('Volatility (Std. Deviation)')
plt.ylabel('Expected Returns')
plt.title('Risk-Return Profiles of Various Assets')
plt.show()
