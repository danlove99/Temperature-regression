import pandas as pd 
import numpy as np  
import tensorflow as tf 
from tensorflow.keras.layers import Dense, Dropout

# Load in CSV

df = pd.read_csv('city_temperature.csv', sep=',', error_bad_lines=False, index_col=False, dtype='unicode')

# Preprocessing

y = df['AvgTemperature'].fillna(0)#.values
df = df.drop(['Region', 'State', 'Country', 
			'Day', 'Year', 'AvgTemperature'], axis=1)
df = pd.get_dummies(df, dummy_na=True).fillna(0)#.values

# remove last row for prediction test
test = df.tail(1)
df = df[:-1]
test_y = y.tail(1)
y = y[:-1]

# model definition

model = tf.keras.models.Sequential()
model.add(Dense(135, input_shape=(135,), activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='Adam', loss=tf.keras.losses.MeanSquaredError(), metrics=['mse'])
model.fit(df, y, batch_size=10, epochs=3)

# print true result and predicted result
result = model.predict(test)
print("Predicted result: {} \nActual result: {}".format(result, test_y))
