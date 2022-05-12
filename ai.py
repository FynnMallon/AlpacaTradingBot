import pandas as pd
from sklearn.model_selection import train_test_split
import keras 
df = pd.read_csv('StockData.csv')
X = pd.get_dummies(df.drop(['Increase'], axis=1))
y = df['Increase'].apply(lambda x: 1 if x=='1' else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
y_train.head()

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score

model = Sequential()
model.add(Dense(units=32, activation='relu', input_dim=len(X_train.columns)))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='sgd', metrics='accuracy')

