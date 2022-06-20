#Trading
from alpaca_trade_api.rest import REST, TimeFrame
from matplotlib.font_manager import json_load
from matplotlib.style import available
from pandas import json_normalize

api = REST(
    'PK5HIQ7ZY7A021X46J13',
    'k6pnSUYmwbCMV9uBqHDk3w2JR875qrIGyCdnYps4',
    'https://paper-api.alpaca.markets') #API Data

account = api.get_account()

#Tensorflow
import tensorflow as tf #deep learning library
from tensorflow import keras #additional, easier to use features, built ontop of and integrated with tensorflow
from keras.layers import Activation, Dense #Used in the depth of the AI
from keras.metrics import categorical_crossentropy #A type of analysis
from keras.models import Sequential #The deep learning model i am using
# from keras.optimizers import Adam  # Works on windows
from keras.models import load_model #Loads previously saved model
import os.path 
from tensorflow.keras.optimizers import Adam   # Works on mac
import math

#other Libraries
import numpy as np
Portfolio = {} #Stock, Shares

def Model():
    if os.path.isfile('models/TSLA_AI.h5') is True:
        model = load_model('models/TSLA_AI.h5')
        print("Loaded Model")
    model.compile(optimizer=Adam(learning_rate=0.0001),
                loss = 'sparse_categorical_crossentropy',
                metrics = ['Accuracy'])
    return Model

def Trade(Call, Stock):
    # available = account.buying_power
    # price = int(api.get_bars(Stock,TimeFrame.Minute, None, None ,adjustment='raw', limit= None)[0].c) #Gets raw bars from Alpaca API Market servers
    if Call == 1:
        api.submit_order(
            symbol = Stock,
            qty = math.trunc((int(available)/price)),
            side = 'buy',
            type = 'market',
            time_in_force= 'gtc'
        )
        print("DOne")
    else:
        stock_position = api.get_position(Stock)
        api.submit_order(
            symbol = Stock,
            qty = stock_position.qty,
            side = 'sell',
            type = 'market',
            time_in_force= 'gtc'
        )
        print("sold")

def Predict(LiveData,TrainingSize):
    predictions = model.predict(x = LiveData, batch_size = TrainingSize, verbose = 0)
    rounded_predictions = np.argmax(predictions, axis=-1)
    return rounded_predictions()

model = Model()
Predict(data, 5)



Trade(0,"AAPL")