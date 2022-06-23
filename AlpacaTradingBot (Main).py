#Trading
from ast import While
from multiprocessing.shared_memory import SharedMemory
from alpaca_trade_api.rest import REST, TimeFrame
from matplotlib.font_manager import json_load
from matplotlib.style import available
from pandas import json_normalize
import os
#Tensorflow
import tensorflow as tf #deep learning library
from tensorflow import keras #additional, easier to use features, built ontop of and integrated with tensorflow
from keras.layers import Activation, Dense #Used in the depth of the AI
from keras.metrics import categorical_crossentropy #A type of analysis
from keras.models import Sequential #The deep learning model i am using
from keras.optimizers import Adam  # Works on windows
from keras.models import load_model #Loads previously saved model
import os.path 
# from tensorflow.keras.optimizers import Adam   # Works on mac
import math
import csv 
#other Libraries
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' #Helps reduce errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' #Hides terminal warnings on program start

import pytz
import datetime
from datetime import datetime, timedelta
import datetime as dt
import time

timeZ_Ny = pytz.timezone('America/New_York')


dt_Ny = datetime.now(timeZ_Ny)
api = REST(
    'PK16R9S2JFF6G1QWJM7Q',
    'ZiNeSYlXMsIFcnVVqNpS7M8CgN3EfAhElkPBdKA7',
    'https://paper-api.alpaca.markets'
    ) #API Data

account = api.get_account()


def Model(): #Gets AI Model
    if os.path.isfile('models/TSLA_AI.h5') is True: #checks if AI model exists
        Model = load_model('models/TSLA_AI.h5') #loads model
        print("Loaded Model")
    else:
        Model = Sequential([
            Dense(units=7, input_shape=(7,), activation='relu'),
            Dense(units=32, activation = 'relu'),
            Dense(units=16, activation = 'relu'),
            Dense(units=2, activation='softmax')
        ]) #creates model if one not present
        print("Established new model")
        
    Model.compile(optimizer=Adam(learning_rate=0.0001), #compiles model
                loss = 'sparse_categorical_crossentropy',
                metrics = ['Accuracy'])
    
    return(Model)

def Trade(Call, Stock, Price):
    available = account.buying_power #Checks account balance
    available = available/10 #trades with 10% available
    if Call == 1:
        api.submit_order(
            symbol = Stock,
            qty = (math.trunc((int(available)/Price))),
            side = 'buy',
            type = 'market',
            time_in_force= 'gtc'
        )
        print("Buy Order placed at {Price}".format(Price =Price))
    else:
        stock_position = api.get_position(Stock)
        api.submit_order(
            symbol = Stock,
            qty = stock_position.qty,
            side = 'sell',
            type = 'market',
            time_in_force= 'gtc'
        )
        print("Sell Order placed at {Price}".format(Price =Price))

def LivePredict(model):
    UTC = pytz.timezone('UTC') #Sets timezone
    time_now = dt.datetime.now(tz=UTC) #Gets current time
    time_15_min_ago = time_now - dt.timedelta(minutes=16) #Sometimes if using -15 it will say subscription invalid, using 16min historic bars to be safe
    time_20_min_ago = time_now - dt.timedelta(minutes=21)
    #Have to delay information by 15mins otherwise have to pay $100USD per month
    LiveData = []
    closes = []
    bars = api.get_bars("TSLA",TimeFrame.Minute, start=time_20_min_ago.isoformat(), 
                end=time_15_min_ago.isoformat(), adjustment='raw', limit= None) #Gets raw bars from Alpaca API Market servers
    for bar in bars:
        data = []
        barinfo = bar.o, bar.c, bar.h, bar.l, bar.n, bar.v, bar.vw #Gets all the bar values
        closes.append(bar.c) #saves close price
        for detail in barinfo:
            data.append(detail) #adds all the bat values to one array
        LiveData.append(data)

    LiveData = np.array(LiveData) #numpifies the array
    if len(LiveData)!=0: #market is closed when length = 0
        predictions = model.predict(x = LiveData, batch_size = 10, verbose =0) #Predicts next bar
        rounded_predictions = np.argmax(predictions, axis=-1) #rounds to 0 or 1
        Trade(rounded_predictions[0],"TSLA",closes[0]) #tradces
    else:
        print("Market is Closed")

model = Model() #Enstantiates Model

while True: #runs minutely
    LivePredict(model) 
    time.sleep(60)



#All used for stub code
def Predict(LiveData,TrainingSize,model):
    predictions = model.predict(x = LiveData, batch_size = TrainingSize, verbose = 0) #predicts
    rounded_predictions = np.argmax(predictions, axis=-1) #rounds to 0 or 1
    return (rounded_predictions) #returns

def ShapeData(file):
    Stub_Data = [] #holds data
    Stub_Prices =[] #holds prices
    with open(file) as csvfile:
        reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) #reads from CSV file
        for line in reader:
            Stub_Prices.append(line[0]) #Stub prices to measure profit ability
            Stub_Data.append(line) 
    Stub_Data = np.array(Stub_Data) #numpifies data
    return(Stub_Data,Stub_Prices)

def StubTrading(Predictions, Prices):
    Balance = 100 #Fake balance
    Stocks = 0 #Fake portfolio
    for i in range (0, len(Predictions)-1): #Runs through each prediction
        if Predictions[i] == 1: #if increase max buy
            Stocks += Balance/Prices[i] #buy stocks at price
            Balance = 0 #balance 0
        else:
            Balance += Prices[i]*Stocks #sell stocks
            Stocks = 0 
    Balance += Prices[i]*Stocks #sells leftover stocks
    Stocks = 0
    Profit = Balance - 100 #profit calculation
    return(Profit)

def StubCode(Model): #Used for when the market is offline
    Data, Prices = ShapeData('TSLA_StubData.csv') #loads stub values
    Predictions = Predict(Data, 5,model) #runs predictions
    print(Predictions) #prints predictions
    print("Profit = " + str(StubTrading(Predictions,Prices))) #prints profit from trades

