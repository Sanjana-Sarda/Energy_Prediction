from kubernetes import client, config

config.load_kube_config()
v1 = client.CoreV1Api()
# v1.list_node() - kubectl get nodes
# v1.list_namespace() - kubectl get namespaces
#v1.list_pod_for_all_namespaces()
#v1.list_persistent_volume_claim_for_all_namespaces()