from tensorflow.keras.models import load_model, Model

model = load_model("autoencoder.h5", compile=False)

encoder = Model(model.input, model.layers[1].output)

decoder_input = model.layers[1].output
decoder = Model(decoder_input, model.layers[2](decoder_input))

encoder.save("encoder.h5")
decoder.save("decoder.h5")

print("Split Done")