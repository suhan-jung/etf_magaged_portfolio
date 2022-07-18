import xlwings as xw

path = r'imdt_auction.xlsm'
wb = xw.Book(path)
getPrice = wb.macro('GetPrice')

print(getPrice("005930"))
