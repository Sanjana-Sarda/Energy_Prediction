from keras.models import load_model
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

print("Running inference...")
NN_model_main = load_model("NN_test.h5")
data = pd.read_csv('test_data_8156.csv')
data = data.drop(columns=['dataid'])
scaler = StandardScaler()
num_points = 336

for d in range (2):
    d_temp = data.iloc[d*num_points:(d+1)*num_points].to_numpy()
    d_temp = scaler.fit_transform(d_temp)