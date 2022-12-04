# Household Energy Prediction using Federated Learning 

Setup tailscale on controller and all nodes. 

On Controller
```
git pull
docker images //Docker Clean up
docker image rm --force <image id> //Docker Clean up
docker build -t k8-model -f Dockerfile .
docker save --output k8-model-latest.tar k8-model:latest
kubectl label nodes <node-name> house=<house-letter>
rsync -v k8-model-latest.tar <node-ip>:k8-model-latest.tar
kubectl create -f inference.yaml
```

Mosquitto Setup
```
sudo apt install mosquitto 
sudo service mosquitto start // for testing
sudo service mosquitto stop
sudo mosquitto -c mosquitto.conf 

```

On Node
```
sudo tailscale up -ssh
sudo apt-get install rsync
sudo k3s ctr images import k8-model-latest.tar 
```