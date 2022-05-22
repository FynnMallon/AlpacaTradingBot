'''
# expects 2 inputs x and why
# x is a numpy array, tensorflow tensor, dict map or tf data
# y contains the corresponding labels for our data and needs to be in the same format as x
'''
from cgi import test
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
#https://www.youtube.com/watch?v=qFJeN9V1ZsI
import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

train_labels = []
train_samples = []

#creating the data ourselves for learning purposes
'''
Example Data:
-an experimental drug is tested on individuals aged 13 to 100
-the trial had 2100 participants
-half under 65, half over
-95% of older pacients had side effects
-95% of patients under did not'''

for i in range(50):
    #5% younger no side effect
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(1)
    
    #5% older no side effect
    random_older = randint(65,100)
    train_samples.append(random_older)
    train_labels.append(1)
    
for i in range(950):
    #95% younger no side effect
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(0)
    
    #95% older no side effect
    random_older = randint(65,100)
    train_samples.append(random_older)
    train_labels.append(0)
    
train_labels = np.array(train_labels)
train_samples = np.array(train_samples)
#Sample Data now converted to a numpy array (Accepted by tensorflow)
train_labels, train_samples = shuffle(train_labels,train_samples)
#Shuffles Data (keys linked) to remove any imposed order

scaler = MinMaxScaler(feature_range=(0,1))
#converts the data to a float between 0-1
#This is much more efficent for the AI
scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1,1))
#Rescales data from a scale of 13-100, down to 0-1
#Reshapes data since it doesnt accept 1D data

#Data is now processed and ready to be accepted by AI
#Timestamp 18:40

#Simple tf.keras sequential Model
import tensorflow as tf
from tensorflow import keras 
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy

model = Sequential([
    Dense(units=16, input_shape=(1,), activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=2, activation = 'softmax')

])

model.summary()


sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))



#Lesson 2 complete, 24:32

model.compile(optimizer=Adam(learning_rate =0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['Accuracy'])

model.fit(x=scaled_train_samples,
          y=train_labels,
          validation_split=0.1,
          batch_size=10,
          epochs=30,
          shuffle=True,
          verbose=2)

#Lesson 3 complete, 30:06
#adding a validation set to test model
#validation_split occurs before model is shuffled in model.fit (validation is the last x %)

#lesson 4 complete, 39:40
test_labels = []
test_samples = []

for i in range(10):
    #5% younger no side effect
    random_younger = randint(13,64)
    test_samples.append(random_younger)
    test_labels.append(1)
    
    #5% older no side effect
    random_older = randint(65,100)
    test_samples.append(random_older)
    test_labels.append(1)
    
for i in range(200):
    #95% younger no side effect
    random_younger = randint(13,64)
    test_samples.append(random_younger)
    test_labels.append(0)
    
    #95% older no side effect
    random_older = randint(65,100)
    test_samples.append(random_older)
    test_labels.append(0)
    
test_labels = np.array(test_labels)
test_samples = np.array(test_samples)
test_labels, test_samples = shuffle(test_labels,test_samples)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_test_samples = scaler.fit_transform(test_samples.reshape(-1,1))

predictions = model.predict(x=scaled_test_samples, batch_size=10, verbose=0)

for i in predictions:
    print(i)
    
rounded_predictions = np.argmax(predictions, axis=-1)
for i in rounded_predictions:
    print(i)