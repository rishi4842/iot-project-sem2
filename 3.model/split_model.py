from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input

# Load model
model = load_model("model/autoencoder.h5", compile=False)

# -------- ENCODER --------
encoder = Model(inputs=model.input, outputs=model.layers[1].output)

# -------- DECODER --------
encoded_input = Input(shape=model.layers[1].output.shape[1:])
x = encoded_input

for layer in model.layers[2:]:
    x = layer(x)

decoder = Model(inputs=encoded_input, outputs=x)

# -------- SAVE --------
encoder.save("model/encoder.h5")
decoder.save("model/decoder.h5")

print("Split Done Successfully ✅")