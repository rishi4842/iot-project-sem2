import pandas as pd
import numpy as np

data = {
    "temp": np.random.uniform(20, 35, 2000),
    "humidity": np.random.uniform(40, 80, 2000),
    "light": np.random.uniform(100, 500, 2000)
}

df = pd.DataFrame(data)
df.to_csv("sensor_data.csv", index=False)

print("Done")