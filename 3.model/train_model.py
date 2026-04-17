import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Generate sample IoT data (simulate sensor values)
data = np.random.randint(0, 100, size=(1000, 3))

# Normalize
data = data / 100.0

# Autoencoder
input_layer = Input(shape=(3,))
encoded = Dense(2, activation='relu')(input_layer)
decoded = Dense(3, activation='sigmoid')(encoded)

autoencoder = Model(input_layer, decoded)
encoder = Model(input_layer, encoded)

# Decoder model
encoded_input = Input(shape=(2,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))

# Compile
autoencoder.compile(optimizer='adam', loss='mse')

# Train
autoencoder.fit(data, data, epochs=100, batch_size=16, verbose=1)

# Save models
encoder.save('../3.model/encoder.h5')
decoder.save('../3.model/decoder.h5')

print("Model retrained and saved!")
