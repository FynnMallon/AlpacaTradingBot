# Alpacca to feed historical data 
# Set Dataset initially to 

# Options for AI to buy sell hold @ strike price
# If/When strike price reached order placed
# AI generational progression based on Profit/Loss

# Use Tensor / Machine Learning
# Unsupervised Learning
# Ai predicts if next dataset will be up or down

# For linear algebra
from fileinput import filename
import numpy as np
# For data processing
import pandas as pd

import scipy as sc
import json
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import REST, TimeFrame
import sklearn
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas
import csv

api = REST(
    'PK1GNWZ4WG3CFDX64EY0',
    'E3GbOVdeLp5GR6HgOW0dxmzZtqsicsBn7DcXNUFz',
    'https://paper-api.alpaca.markets'
)

def get_bars():
    # data = open('data.txt', 'w')
    bars = api.get_bars("IBM",TimeFrame.Minute, "2021-06-14", "2021-06-15", adjustment='raw', limit= None)
    filename = "StockData.csv"
    csvfields = ['Open','Close','High','Low','Number of Trades','Volume','Volume Weighted Average','Increase']
    csvrows = []
    for bar in bars:
        # Timestamp = bar.t
        # Timestamp = str(Timestamp)
        barinfo = bar.o, bar.c, bar.h, bar.l, bar.n, bar.v, bar.vw
        # Trade = {
        # # 'Time' : Timestamp,
        # 'Open' : bar.o,
        # 'Close' : bar.c,
        # 'High' : bar.h,
        # 'low' : bar.l,
        # 'Number of Trades' : bar.n,
        # 'Volume' : bar.v,
        # 'Volume-Weighted Average' : bar.vw,
        # }
        csvrows.append(barinfo)
    # data.close()
    print(csvrows)
    with open(filename,'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(csvfields)
        csvwriter.writerows(csvrows)
        
def ai():
    iris_df = datasets.load()
    print(dir(iris_df))
    # Features
    print(iris_df.data)

    # Targets
    print(iris_df.filename)

    # Target Names
    print(iris_df.target_names)
    label = {0: 'red', 1: 'blue', 2: 'green'}

get_bars()
