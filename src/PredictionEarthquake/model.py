import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
%matplotlib inline
import datetime 
import time
from sklearn.model_selection import train_test_split
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

sns.set_style('darkgrid')
## reading the file from directort

df = pd.read_csv('database.csv')
## display top 5 rows

df.head()
df.shape
## main features that are required to predict earthquake

df = df[['Date', 'Time', 'Latitude', 'Longitude', 'Depth', 'Magnitude']]
df.head()
## checking if there is any null values or not

df.isnull().sum()
# converting Date column from string to datetime

df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format='%m/%d/%Y')
## Extracting minutes, seconds, hours from column Time

minutes = []
hours = []
seconds = []

for t in df.Time:
    t  = t.split(":")
    minutes.append(t[1])
    hours.append(t[0])
    seconds.append(t[2])
## creating new columns minutes, hours, seconds

df['minutes'] = minutes
df['hours'] = hours
df['seconds'] = seconds

df['minutes'] = pd.to_numeric(df['minutes'], errors='coerce')
df['hours'] = pd.to_numeric(df['hours'], errors='coerce')
df['seconds'] = pd.to_numeric(df['seconds'], errors='coerce') 
## creating new columns year, month, day

df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day
## dropping date and time column from dataset

df.drop(['Date', 'Time'], axis=1, inplace=True)
df.head()
df.dropna(inplace=True)
df.isnull().sum()
## independent and dependent features

X = df[['Latitude', 'Longitude', 'minutes', 'hours', 'seconds', 'year', 'month', 'day']]
y = df[['Depth', 'Magnitude']]
X = X.values
y = y.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
## creating a neural Network model

model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(8,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='adam', loss='squared_hinge', metrics=['accuracy'])
## training the model

history = model.fit(X_train, y_train, batch_size=64, epochs=20, verbose=1, validation_data=(X_test, y_test))
