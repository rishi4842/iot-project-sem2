import os
from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
encoder = load_model(os.path.join(BASE_DIR, '..', '3.model', 'encoder.h5'), compile=False)
decoder = load_model(os.path.join(BASE_DIR, '..', '3.model', 'decoder.h5'), compile=False)

@app.route('/', methods=['GET', 'POST'])
def home():
    original = []
    compressed = None
    reconstructed = None
    error = None
    message = None

    if request.method == 'POST':
        data = request.form.get('data', '')

        try:
            original = [float(x) for x in data.split(',') if x.strip() != '']
            if len(original) == 0:
                raise ValueError('No numeric values provided.')

            values = np.array(original).reshape(1, -1)
            values = values / 100.0

            compressed = encoder.predict(values)
            reconstructed = decoder.predict(compressed)
            reconstructed = reconstructed * 100.0

            error = float(np.mean((np.array(original).reshape(1, -1) - reconstructed) ** 2))

            compressed = compressed.flatten().tolist()
            reconstructed = reconstructed.flatten().tolist()

        except Exception as e:
            print('ERROR:', e)
            message = 'Invalid input. Please enter comma-separated numbers like 30,45,60.'

    return render_template(
        'index.html',
        original=original,
        compressed=compressed,
        reconstructed=reconstructed,
        error=error,
        message=message
    )
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5050)