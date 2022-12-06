from keras.models import load_model
import pandas as pd
from sklearn.metrics import mean_absolute_error 
from sklearn.preprocessing import StandardScaler 
import numpy as np
import paho.mqtt.client as mqtt
import paho.mqtt.publish as publish 
import paho.mqtt.subscribe as subscribe
import tensorflow as tf
from keras.optimizers import Adam
import pickle
import os


def on_message(clientdata, userdata, msg):
    global model_received, weights
    print("Subscribed")
    weights = deserialize(msg.payload)
    model_received = True

def on_publish(clientdata, userdata, msg):
    print("Published")


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
house_name = "d"

mqttBroker = "100.90.105.93"

client = mqtt.Client("House_"+house_name)
client.on_message = on_message
client.on_publish = on_publish
client.connect(mqttBroker)
client.subscribe("Global_Model")


print("Running inference...")
NN_model = load_model("NN_test.h5")
data = pd.read_csv('test_data.csv', index_col=0)
data = data.reset_index(drop=True)
scaler_mean = pd.read_csv('mean_orig.csv').iloc[:, 1].to_numpy().reshape((1, 1450))
scaler_var = pd.read_csv('var_orig.csv').iloc[:, 1].to_numpy().reshape((1, 1450))
scaler = StandardScaler()
data = data.drop(columns=['dataid'])

y = data[['Total Energy (kWh)', 'Total Solar Energy (kWh)']]
X = data.drop(columns=['Total Energy (kWh)', 'Total Solar Energy (kWh)'])
X_week = np.zeros([num_points, X.shape[1]])
y_week = np.zeros([num_points, y.shape[1]])

for idx, X_row, in X.iterrows():
    X_row = np.array(X_row).reshape((1, 1450))
    y_row = np.array(y.iloc[idx])
    X_week[count] = X_row
    y_week[count] = y_row
    X_row = (X_row - scaler_mean)/np.sqrt(scaler_var)
    inf = NN_model(X_row)
    all_inf.append(str(float(inf[0, 0]))+"\t"+str(float(inf[0, 1])))

    
    count +=1
    if (count == num_points):
        X_week = scaler.fit_transform(X_week)
        pre_mae = mean_absolute_error(NN_model(X_week), y_week)
        print ("pre_mae ="+ str(pre_mae))
        trained_model = tf.keras.models.clone_model(NN_model)
        trained_model.set_weights(NN_model.get_weights())
        trained_model.set_weights(train(trained_model, X_week, y_week))
        post_mae = mean_absolute_error(trained_model(X_week), y_week)
        print ("post_mae ="+ str(post_mae))
        if (post_mae<=pre_mae):
            NN_model.set_weights(trained_model.get_weights())
        else:
            post_mae = pre_mae
        #NN_model.set_weights(train(NN_model, X_week, y_week))
        #post_mae = mean_absolute_error(NN_model(X_week), y_week)
        publish.single("House/model/"+house_name, serialize(NN_model), hostname = mqttBroker) 
        publish.single("House/pre_mae/"+house_name,  pre_mae, hostname = mqttBroker) 
        publish.single("House/post_mae/"+house_name,  post_mae, hostname = mqttBroker) 
        client.loop_start()
        while(not model_received):
            print ("waiting")
            continue
        client.loop_stop()
        #weights = deserialize(subscribe.simple("Global_Model", hostname =mqttBroker, keepalive=60).payload)
        updated_model = tf.keras.models.clone_model(NN_model)
        updated_model.set_weights(weights)
        updated_model.set_weights(finetune(updated_model, X_week, y_week))
        updated_mae = mean_absolute_error(updated_model(X_week), y_week)
        print ("updated_mae" + str(updated_mae))
        if (updated_mae<=post_mae):
            NN_model.set_weights(updated_model.get_weights())
        scaler_mean = scaler.mean_
        scalar_var = scaler.var_
        X_week = np.zeros([num_points, X.shape[1]])
        y_week = np.zeros([num_points, y.shape[1]])
        count = 0 
publish.single("House/inf/"+house_name, pickle.dumps(all_inf), hostname = mqttBroker) 