import requests
import numpy as np
from tensorflow.keras.models import load_model

encoder = load_model("model/encoder.h5", compile=False)

sample = np.array([[30, 60, 200]])

compressed = encoder.predict(sample).tolist()

requests.post("http://127.0.0.1:5000/data", json={"data": compressed})

print("Sent:", compressed)