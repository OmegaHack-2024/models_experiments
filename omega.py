import numpy as np
import pandas as pd

data = pd.read_csv("consumo_casa.csv")
data
data['Fecha'] = pd.to_datetime(data['Fecha'])
print(data['Fecha'].dt.minute)