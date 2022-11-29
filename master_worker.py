from kubernetes import client, config

config.load_kube_config()
v1 = client.CoreV1Api()
print (v1.list_node())