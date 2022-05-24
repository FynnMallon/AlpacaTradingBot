import csv
import os
import random

import numpy as np #mathamatical library
from matplotlib.style import use
from sklearn.preprocessing import MinMaxScaler #Used for converting intigers to a float between 0-1
from sklearn.utils import shuffle #Shuffles two dictionaries while preserving the link between their indexes
import time


def EstablishTrainingData(file):
    Training_Data = []
    Training_Labels = []

    #Reads StockData From CSV file into Python Array
    with open(file) as csvfile:
            reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
            for line in reader:
                Training_Data.append(line)

    #If the Stock Increases from one minute to the next, It is Labeled '1', if it Decreases labeled '0'
    for Entry in range(0,len(Training_Data)-1):
        if Training_Data[Entry]<Training_Data[Entry+1]:
            Training_Labels.append(1)
        else:
            Training_Labels.append(0)  
    Training_Labels.append(0) #writes for final line (need both to be the same size)
    
    #Converts Data into a Numpy Array (Accepted by TensorFlow)
    Training_Data = np.array(Training_Data)
    Training_Labels = np.array(Training_Labels)

    #Shuffles data relative to eachother to remove any imposed order
    Training_Data, Training_Labels = shuffle(Training_Data,Training_Labels)

    #Should I scale data to a 0-1 range to make it easier to train?
    #Could Raise errors with different fields

    return(Training_Data, Training_Labels)

#Contemplated saving these datasets to a file so i dont have to process them each time, but upon timing it only takes 0.27seconds so it is not worth it


import tensorflow as tf #deep learning library
from tensorflow import keras #additional, easier to use features, built ontop of and integrated with tensorflow
from keras.layers import Activation, Dense #Used in the depth of the AI
from keras.metrics import categorical_crossentropy #A type of analysis
from keras.models import Sequential #The deep learning model i am using
from keras.optimizers import Adam  # Works on windows
from keras.models import load_model #Loads previously saved model
import os.path 
#from tensorflow.keras.optimizers import Adam   # Works on mac



def TrainingModel (Training_Data, Training_Labels):
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' #Helps reduce errors
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' #Hides terminal warnings on program start
    
    
    if os.path.isfile('models/TSLA_AI.h5') is True:
        model = load_model('models/TSLA_AI.h5')
        print("Loaded Model")
        #Checks if there is a saved model and if so, loads it
    else:
        model = Sequential([
            Dense(units=16, input_shape=(7,), activation='relu'),
            Dense(units=32, activation = 'relu'),
            Dense(units=2, activation='softmax')
        ])
        print("Established new model")
        #If code is ran for the first time, a fresh model is generated in this structure

    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
    #allows for running on gpu

    model.compile(optimizer=Adam(learning_rate=0.0001),
                loss = 'sparse_categorical_crossentropy',
                metrics = ['Accuracy'])
    #compiles the model allowing for it to be run, aiming to increase accuracy
    model.fit(x =Training_Data, #The data itself
            y = Training_Labels, #The labelled corrosponding data
            validation_split=0.1, #Splits 10% of the data for validation allowing to see if it is beeing badly overfitted
            batch_size=100, #Each generation goes through 100 datasets
            epochs=100, #The ammount of generations
            shuffle=True, #Again Shuffles input (Note this is done after validation split so it is still important to have my shuffling in the data formation)
            verbose=2 )
    #Runs on the training Data
    
    model.save('models/TSLA_AI.h5') #Saves the progress


for i in range(0,5): #Shuffles data 5 times, helps reduce BIAS
    Training_Data, Training_Labels = EstablishTrainingData('TSLA_Data.csv')
    TrainingModel(Training_Data, Training_Labels)