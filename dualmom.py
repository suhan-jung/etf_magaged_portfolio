import pandas as pd
import math
import util
import sys

""" TODOs
-------------------------------------------------------------------------------------"""

# O. update universe (exclude survivorship bias)

# O. reduce portfolio volatility
#    - MPN ?
#    - model parameters (more diversification)

""" ETC
-------------------------------------------------------------------------------------"""

# logging
LOGFILENAME = "logfile"
log = open(LOGFILENAME, "a")
sys.stdout = log


""" 0 - DATA IN
-------------------------------------------------------------------------------------"""

# 0-0 : kospi200.csv kosdaq150.csv
FILENAME_IN = "ETF.csv"         # SNP500_CLEAN.csv 
FILENAME_IN = "data_pp_20220627"         # SNP500_CLEAN.csv 
data_df = pd.read_csv(            # data : pandas dataframe 
    FILENAME_IN,
    index_col = 0
).astype(float)

# 0-1 : delete columns if 2000-01-03 data is not available
for col in data_df.columns :
    if data_df[col].isnull().any() == True :
        data_df.drop(col, axis = 1, inplace = True)

# 0-2 : add CASH as an asset with 0 return
data_df['CASH'] = 0

# data_df = data_df.iloc[2515:] # 2515 : 2010-01-02

""" 1 - DATA
-------------------------------------------------------------------------------------"""

# 1-1 : copy dataframe for cumulative returns
data_df_cum = pd.DataFrame(
    index = data_df.index,
    columns = data_df.columns
)

# set row 0 : 1
cols = data_df.columns
for col in cols :
    data_df_cum.iloc[0][col] = 1

# set rows 1 ~ end : cumulative return
for index, row in data_df.iterrows() :
    
    # print('index : ' + str(index))

    if index == data_df_cum.index[0] :
        prev_row = data_df_cum.iloc[0]
    
    else :
    
        for col in cols : # update columns
            val                        = prev_row[col] * (1 + data_df.loc[index][col])
            data_df_cum.at[index, col] = val

    prev_row = data_df_cum.loc[index]

data_df_cum.reset_index(inplace = True)


""" 2 - DATA - other variables / print
-------------------------------------------------------------------------------------"""

# vars & etc
columns         = list(data_df.columns)

# print
dftype = type(data_df)

print(dftype)


""" 3 - MODEL
-------------------------------------------------------------------------------------"""

LOOKBACK_1             = 250            # 250 days
LOOKBACK_2             = 125            # 125 days
LOOKBACK_3             = 65             # 65  days 
# NUM_ASSETS             = 20             # 20  max number of assets in portfolio
NUM_ASSETS             = 5             # 20  max number of assets in portfolio
UPDATE_FREQ            = 22             # 22  rebalance every quarter

sta                    = 0
end                    = data_df_cum.index[-1]
pf_initial             = 100
pf_values              = [pf_initial]
win_count              = 0

# iterate through rows of cumulative returns dataframe
for iter in range(sta + LOOKBACK_1, end) : # 0 ~ 2800

    ## A. get lookback-returns for each asset
    row_cur = data_df_cum.iloc[iter]              # current
    row_a   = data_df_cum.iloc[iter - LOOKBACK_1] 
    row_b   = data_df_cum.iloc[iter - LOOKBACK_2] 
    row_c   = data_df_cum.iloc[iter - LOOKBACK_3] 

    lookback_returns = []
    for col in columns :

        ## (return, rank) of assets
        lr1 = float(row_cur[col] / row_a[col] - 1) # lr : lookback-return
        lr2 = float(row_cur[col] / row_b[col] - 1) # lr : lookback-return
        lr3 = float(row_cur[col] / row_c[col] - 1) # lr : lookback-return

        lr1_ann = lr1 * (250.0/LOOKBACK_1)
        lr2_ann = lr2 * (250.0/LOOKBACK_2)
        lr3_ann = lr3 * (250.0/LOOKBACK_3)

        lr = (lr1_ann + lr2_ann + lr3_ann) / 3.0
        lookback_returns.append(lr)

    ## B. rebalancing every UPDATE_FREQ days
    if (iter % UPDATE_FREQ == 0) or (iter == sta + LOOKBACK_1) :    

        assets  = []
        returns = []

        for assetiter in range(0, NUM_ASSETS, 1) :
           
            asset_pos    = util.nthlargest(lookback_returns, assetiter + 1)
            asset_name   = columns[asset_pos]
            asset_return = util.nthlargest_val(lookback_returns, assetiter + 1)

            if (asset_return <= 0.0) : # or (~~)
                print('continue : ' + str(asset_name))
                continue

            assets.append(asset_name)
            returns.append(asset_return)

        tostring = "[REBALANCE @ " + str(data_df.index[iter]) + "] "
        nasset   = len(assets)
        for assetiter in range(0, nasset, 1) :    

            tostring = tostring + str(assets[assetiter]) + " : " + str(returns[assetiter]) + " / "

        print(tostring)

        if nasset == 0 :
            assets = ['CASH']
            nasset = 1

    ## C. compute daily portfolio return / value
    ret          = 0
    tostring = "[VALUE UPDATE @ " + str(data_df.index[iter]) + "] "
    for assetiter in range(0, nasset, 1) :

        assetname   = assets[assetiter] # assetname
        assetreturn = data_df.iloc[iter][assetname]
        ret        += assetreturn
        tostring    = tostring + str(assets[assetiter]) + " : " + str(assetreturn * 100)  + " / "

    print(tostring)
    
    ret /= nasset
    new_pfvalue = pf_values[-1] * (1 + ret)
    pf_values.append(new_pfvalue)
    if ret > 0 :
        win_count += 1

    print('pf. return : ' + str(ret))
    print('pf. new value : ' + str(new_pfvalue))

    # print(tostring)

""" PRINT & PLOT
-------------------------------------------------------------------------------------"""

# plot pf. performance graph
from matplotlib import pyplot as plt
plt.plot(pf_values)
plt.show()

# write pf. performance to file
with open('performance.log','w') as f :
    for elem in pf_values :
        f.write("%s\n" % elem)

print('win_count : ' + str(win_count))