# import pandas as pd
# from sklearn.model_selection import train_test_split
# from tensorflow import keras 
# import os

# # os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# df = pd.read_csv('NewData.csv')
# X = pd.get_dummies(df.drop(['Increase'], axis=1))
# y = df['Increase']
# print(y)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
# y_train.head()

# from keras.models import Sequential, load_model
# from keras.layers import Dense
# from sklearn.metrics import accuracy_score

# model = Sequential()
# model.add(Dense(units=32, activation='relu', input_dim=len(X_train.columns)))
# model.add(Dense(units=64, activation='relu'))
# model.add(Dense(units=1, activation='sigmoid'))

# model.compile(loss='mean_squared_error', optimizer='adam', metrics='accuracy')
# model.fit(X_train, y_train, epochs=200, batch_size=16)
# y_hat = model.predict(X_test)
# y_hat = [0 if val < 0.5 else 1 for val in y_hat]
# accuracy_score(y_test, y_hat)