
from genericpath import exists
from numbers import Real
import tensorflow as tf #deep learning library
from tensorflow import keras #additional, easier to use features, built ontop of and integrated with tensorflow
from keras.layers import Activation, Dense #Used in the depth of the AI
from keras.metrics import categorical_crossentropy #A type of analysis
from keras.models import Sequential #The deep learning model i am using
# from keras.optimizers import Adam  # Works on windows
from keras.models import load_model #Loads previously saved model
import os.path 
from tensorflow.keras.optimizers import Adam   # Works on mac

from fileinput import filename
import numpy as np
# For data processing
import pandas as pd

import scipy as sc
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import REST, TimeFrame
import csv

api = REST(
    'PKUCCTWFOPIZBLWK9D3Q',
    'FM5ipVhzyQosDDUKRs3BUwZBi4SEqfW4YGKQ0PLr',
    'https://paper-api.alpaca.markets'
) #API Data

if os.path.isfile('models/TSLA_AI.h5') is True:
    model = load_model('models/TSLA_AI.h5')
    print("Loaded Model")


model.compile(optimizer=Adam(learning_rate=0.0001),
            loss = 'sparse_categorical_crossentropy',
            metrics = ['Accuracy'])

LastBar= None
Prediction = None
Correct = 0
Wrong = 0
LiveData = []
closes = []
bars = api.get_bars("TSLA",TimeFrame.Minute, None, None ,adjustment='raw', limit= None) #Gets raw bars from Alpaca API Market servers
for bar in bars:
    data = []
    barinfo = bar.o, bar.c, bar.h, bar.l, bar.n, bar.v, bar.vw
    closes.append(bar.c)
    for detail in barinfo:
        data.append(detail)
    LiveData.append(data)

LiveData = np.array(LiveData)
print(LiveData)
predictions = model.predict(x = LiveData, batch_size = 10, verbose =0)
rounded_predictions = np.argmax(predictions, axis=-1)
print(rounded_predictions)

RealLabels = []
x=0
for i in range(0,len(closes)-2):
    if closes[i] < closes[i+1]:
        increases = 1
    else:
        increases = 0
    RealLabels.append(increases)

rounded_predictions = list(rounded_predictions.flatten())
for i in range(0,len(RealLabels)-1):
    if RealLabels[i] == rounded_predictions[i]:
        Correct +=1
    else:
        Wrong+=1
print("Correct: " + str(Correct))
print("Wrong: " + str(Wrong))