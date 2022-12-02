from keras.models import load_model
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error 
import numpy as np
import paho.mqtt.publish as publish 
import paho.mqtt.subscribe as subscribe
import tensorflow as tf
from keras.optimizers import Adam


def train(model, X_train, y_train):
    model.compile(optimizer=Adam(1e-5), loss='mean_absolute_error', metrics=['mean_absolute_error'])
    model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split = 0.2)
    return model

def finetune(model, X_train, y_train):
    for l in range(len(model.layers) - 2):
        model.layers[l].trainable = False
        model.compile(optimizer=Adam(1e-2), loss='mean_absolute_error', metrics=['mean_absolute_error'])
        model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split = 0.2)

num_points = 336
count = 0
all_inf = []

mqttBroker = "100.90.105.93"

print("Running inference...")
NN_model = load_model("NN_test.h5")
data = pd.read_csv('test_data_8156.csv', index_col=0)
data = data.drop(columns=['dataid'])

y = data[['Total Energy (kWh)', 'Total Solar Energy (kWh)']]
X = data.drop(columns=['Total Energy (kWh)', 'Total Solar Energy (kWh)'])
scaler = StandardScaler()
X_week = np.zeros([num_points, X.shape[1]])
y_week = np.zeros([num_points, y.shape[1]])

for idx, X_row, in X.iterrows():
    X_row = np.array(X_row).reshape((1, 1450))
    y_row = np.array(y.iloc[idx])
    #X_row = scaler.fit_transform(X_row)
    inf = NN_model(X_row)
    all_inf.append(inf)
    X_week[count] = X_row
    y_week[count] = y_row
    
    count +=1
    if (count == num_points):
        pre_mae = mean_absolute_error(NN_model(X_week), y_week)
        NN_model = train(NN_model, X_week, y_week)
        post_mae = mean_absolute_error(NN_model(X_week), y_week)
        #publish.single("House/a", [NN_model, pre_mae, post_mae], hostname = mqttBroker) # send mae before and after
        publish.single("House/a",  pre_mae, hostname = mqttBroker) # send mae before and after
        #updated_model = subscribe.simple("Gobal_Model", hostname =mqttBroker, keepalive=60)
        #updated_model = finetune(updated_model, X_week, y_week)
        #updated_mae = mean_absolute_error(updated_model(X_week), y_week)
        #if (post_mae<=updated_mae):
         #   NN_model = tf.keras.models.clone_model(updated_model)
        #X_week = np.zeros([num_points, X.shape[1]])
        #y_week = np.zeros([num_points, y.shape[1]])
        #count = 0 
        
        
    
    