import pandas as pd
from datetime import datetime

def nthlargest(mylist, n) :

    newlist = list(mylist)
    newlist.sort(reverse = True)
    val     = newlist[n-1]

    pos     = mylist.index(val)
    
    return pos

def nthlargest_val(mylist, n) :

    newlist = list(mylist)
    newlist.sort(reverse = True)
    val     = newlist[n-1]
    
    return val

def CONSTITUENTS(FILENAME_INITIAL, FILENAME_JOINERS, FILENAME_LEAVERS, time) :

    # initial
    df_initial = pd.read_csv(
        FILENAME_INITIAL,
        header = None
    )
    
    initial = list(df_initial[df_initial.columns[0]])

    # joiners
    df_joiners = pd.read_csv(
        FILENAME_JOINERS,
        header = None
    )
    
    datecol = df_joiners[df_joiners.columns[0]]
    joiners = df_joiners[df_joiners.columns[1]]
    
    for elem in datecol :
        if datetime.strptime(elem, "%y-%m-%d") < datetime.strptime(time, "%y-%m-%d") :
            pos = datecol.index(elem)

    initial.extend(joiners[:pos])

    # leavers
    df_leavers = pd.read_csv(
        FILENAME_LEAVERS
    )

    # adjusted
    adjusted = initial
    adjusted # remove leavers
     # add joiners
    
    # print
    # print(initial)
    print(df_joiners)

    return adjusted


if __name__ == '__main__' :

    # ll = [1, 3, 2, 4, 100, 5]
    # nn = nthlargest(ll, 3)
    # print(nn)

    # mylist = [0] * 5
    # print(mylist)

    FILENAME_INITIAL = "SNP_INITIAL.csv"
    FILENAME_JOINERS = "SNP_JOINERS.csv"
    FILENAME_LEAVERS = "SNP_LEAVERS.csv"
    time             = "2016-01-01"
    CONSTITUENTS(FILENAME_INITIAL, FILENAME_JOINERS, FILENAME_LEAVERS, time)


    