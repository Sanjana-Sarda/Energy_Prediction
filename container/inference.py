from keras.models import load_model
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error 
import numpy as np
import paho.mqtt.client as mqtt
import paho.mqtt.publish as publish 
import paho.mqtt.subscribe as subscribe
import tensorflow as tf
from keras.optimizers import Adam
import pickle


def train(model, X_train, y_train):
    model.compile(optimizer=Adam(1e-5), loss='mean_absolute_error', metrics=['mean_absolute_error'])
    model.fit(X_train, y_train, epochs=2, batch_size=32, validation_split = 0.2)
    return model.get_weights()

def finetune(model, X_train, y_train):
    for l in range(len(model.layers) - 2):
        model.layers[l].trainable = False
        model.compile(optimizer=Adam(1e-2), loss='mean_absolute_error', metrics=['mean_absolute_error'])
        model.fit(X_train, y_train, epochs=2, batch_size=32, validation_split = 0.2)
    return model.get_weights()

#def on_connect(client, userdata, msg):
 #   client.subscribe("Global_Model")

#def on_message (client, userdata, msg):
 #   global weights, model_received
  #  weights = deserialize(userdata)
   # model_received = True
    
def serialize(model):
    return pickle.dumps({"weights": model.get_weights()})

def deserialize(nn_bytes):
    loaded = pickle.loads(nn_bytes)
    weights = loaded['weights']
    return weights

num_points = 336
count = 0
all_inf = []
model_received = False

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

#client = mqtt.Client("House_a")
#client.on_connect = on_connect
#client.on_message = on_message


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
        NN_model.set_weights(train(NN_model, X_week, y_week))
        post_mae = mean_absolute_error(NN_model(X_week), y_week)
        publish.single("House/a", serialize(NN_model), hostname = mqttBroker) 
        publish.single("House/a/pre_mae",  pre_mae, hostname = mqttBroker) 
        publish.single("House/a/post_mae",  post_mae, hostname = mqttBroker) 
        weights = deserialize(subscribe.simple("Global_Model", hostname =mqttBroker, keepalive=60))
        updated_model = tf.keras.models.clone_model(NN_model)
        updated_model.set_weights(weights)
        updated_model.set_weights(finetune(updated_model, X_week, y_week))
        updated_mae = mean_absolute_error(updated_model(X_week), y_week)
        if (post_mae<=updated_mae):
            NN_model.set_weights(updated_model.get_weights())
        X_week = np.zeros([num_points, X.shape[1]])
        y_week = np.zeros([num_points, y.shape[1]])
        count = 0 
        
        
    
    