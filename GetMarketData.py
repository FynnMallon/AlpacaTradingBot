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
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import REST, TimeFrame
import csv

api = REST(
    'PKR1ED6ACXB64GLNCT0V',
    '9h8HX2Udky4z8H3RGBv0ew3ITuGvdwcfYvDVUIQg',
    'https://paper-api.alpaca.markets'
) #API Data

def get_bars():
    bars = api.get_bars("TSLA",TimeFrame.Minute, "2022-06-19", "2022-06-19", adjustment='raw', limit= 100) #Gets raw bars from Alpaca API Market servers
    filename = "TSLA_StubData.csv" #Filename
    csvrows = [] 
    for bar in bars:
        barinfo = bar.o, bar.c, bar.h, bar.l, bar.n, bar.v, bar.vw #Gets all the bar info
        csvrows.append(barinfo)
    with open(filename,'w') as csvfile:
        csvwriter = csv.writer(csvfile, lineterminator='\n') #writes to csv file
        csvwriter.writerows(csvrows)
    csvfile.close()
        
get_bars()
