
from genericpath import exists
from numbers import Real
import tensorflow as tf #deep learning library
from tensorflow import keras #additional, easier to use features, built ontop of and integrated with tensorflow
from keras.layers import Activation, Dense #Used in the depth of the AI
from keras.metrics import categorical_crossentropy #A type of analysis
from keras.models import Sequential #The deep learning model i am using
from keras.optimizers import Adam  # Works on windows
from keras.models import load_model #Loads previously saved model
import os.path 
# from tensorflow.keras.optimizers import Adam   # Works on mac

from fileinput import filename
import numpy as np
# For data processing
import pandas as pd

import scipy as sc
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import REST, TimeFrame
import csv

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' #Helps reduce errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' #Hides terminal warnings on program start

api = REST(
    'PK16R9S2JFF6G1QWJM7Q',
    'ZiNeSYlXMsIFcnVVqNpS7M8CgN3EfAhElkPBdKA7',
    'https://paper-api.alpaca.markets'
    ) #API Data

if os.path.isfile('models/TSLA_AI.h5') is True: #checks if AI model exists
        model = load_model('models/TSLA_AI.h5') #loads model
        print("Loaded Model")
else:
    model = Sequential([
        Dense(units=7, input_shape=(7,), activation='relu'),
        Dense(units=32, activation = 'relu'),
        Dense(units=16, activation = 'relu'),
        Dense(units=2, activation='softmax')
    ]) #creates model if one not present
    print("Established new model")
    
model.compile(optimizer=Adam(learning_rate=0.0001), #compiles model
            loss = 'sparse_categorical_crossentropy',
            metrics = ['Accuracy'])

LastBar= None
Prediction = None
Correct = 0
Wrong = 0
LiveData = []
closes = []
bars = api.get_bars("TSLA",TimeFrame.Minute, start = "2022-06-19", end = "2022-06-20",adjustment='raw', limit= None) #Gets raw bars from Alpaca API Market servers
for bar in bars:
    data = []
    barinfo = bar.o, bar.c, bar.h, bar.l, bar.n, bar.v, bar.vw
    closes.append(bar.c)
    for detail in barinfo:
        data.append(detail)
    LiveData.append(data) #saves bar info to array

LiveData = np.array(LiveData) #numpifies it 
if len(LiveData) != 0:
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
        RealLabels.append(increases)#measures an increase or a decrease

    rounded_predictions = list(rounded_predictions.flatten()) 
    for i in range(0,len(RealLabels)-1): #compares to AI
        if RealLabels[i] == rounded_predictions[i]:
            Correct +=1
        else:
            Wrong+=1
    print("Correct: " + str(Correct)) #results
    print("Wrong: " + str(Wrong))
else:
    print("Api Error")