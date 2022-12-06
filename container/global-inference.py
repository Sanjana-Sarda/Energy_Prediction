from keras.models import load_model
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error 
import numpy as np
import paho.mqtt.client as mqtt
import paho.mqtt.publish as publish 
import paho.mqtt.subscribe as subscribe
from keras.optimizers import Adam
import pickle

def serialize(model):
    return pickle.dumps({"weights": model.get_weights()})

def deserialize(nn_bytes):
    loaded = pickle.loads(nn_bytes)
    weights = loaded['weights']
    return weights

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        client.subscribe("Global_Model")
        
def get_house_model(client, userdata, msg):
    global model_weights
    model_weights[msg.topic[-1]] = deserialize(msg.payload)
    if (len(model_weights.values())==houses):
        for i in range(6):
            NN_model.weights[i] = sum([weightage[k]*model_weights[k][i] for k in weightage.keys()])
        client.publish("Global_Model", serialize(NN_model))
        model_weights = {}

def get_inf(client, userdata, msg):
    loaded = pickle.loads(msg.payload)
    df=pd.DataFrame.from_dict(loaded)
    df.to_csv("inf_"+msg.topic[-1]+".csv")
    
    
def on_message(clientdata, userdata, msg):
    print("Subscribed")

def on_publish(clientdata, userdata, msg):
    print("Published")

model_weights = {}
weightage = {"a":0.25, "b":0.25, "c": 0.25, "d": 0.25}
houses = 4

NN_model = load_model("NN_test.h5")
    
mqttBroker = "100.90.105.93"

client = mqtt.Client("Controller")

client.on_connect = on_connect
#client.message_callback_add('House/pre_mae', pre_mae)
#client.message_callback_add('House/pre_mae', post_mae)
client.message_callback_add('House/model/#', get_house_model)
client.message_callback_add('House/inf/#', get_inf)
client.on_message = on_message
client.on_publish = on_publish
client.connect(mqttBroker)
client.subscribe("House/#")



    

client.loop_forever()
