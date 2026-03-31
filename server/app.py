from flask import Flask, request
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load decoder model
decoder = load_model("../model/decoder.h5", compile=False)

# API route
@app.route('/data', methods=['POST'])
def receive():
    data = np.array(request.json["data"])
    result = decoder.predict(data)

    print("Reconstructed:", result)

    # Original data (for error calculation)
    original = np.array([[30, 60, 200]])

    # Mean Squared Error
    mse = np.mean((original - result)**2)

    print("Error:", mse)

    return {"status": "ok"}


# Run server (OUTSIDE function)
app.run(debug=True)