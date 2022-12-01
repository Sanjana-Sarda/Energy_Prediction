# Household Energy Prediction using Federated Learning 

Setup tailscale on controller and all nodes. 

On Controller
```
docker build -t k8-model -f Dockerfile .
docker save --output k8-model:latest.tar k8-model:latest
kubectl label nodes <node-name> house=<house-letter>
rsync -v k8-model-latest.tar <node-ip>:k8-model-latest.tar
kubectl create -f inference.yaml
```

On Node
```
sudo tailscale up -ssh
sudo k3s ctr images import k8-model-latest.tar 
```