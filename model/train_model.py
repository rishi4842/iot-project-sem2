import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

data = pd.read_csv("../data/sensor_data.csv").values

input_layer = Input(shape=(3,))
encoded = Dense(2, activation='relu')(input_layer)
decoded = Dense(3)(encoded)

model = Model(input_layer, decoded)
model.compile(optimizer='adam', loss='mse')

model.fit(data, data, epochs=20)

model.save("autoencoder.h5")

print("Model Trained")