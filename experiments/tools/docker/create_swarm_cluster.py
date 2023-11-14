import os
import shutil
import subprocess
import argparse
import time
import json

start = time.time()

SCP_PREFIX = os.environ['SCP_PREFIX'] if 'SCP_PREFIX' in os.environ else "scp"
SSH_PREFIX = os.environ['SSH_PREFIX'] if 'SSH_PREFIX' in os.environ else "ssh"
DS4GNN_MANAGER_IP = os.environ['DS4GNN_MANAGER_IP']
DATASET_PATH = os.environ['DS4GNN_DATASET_PATH']
WORKSPACE_PATH = os.environ['DS4GNN_WORKSPACE_PATH']

CLUSTER_NET_NAME='ds4gnn_net'

opt = subprocess.check_output("whoami", shell=True)
opt = opt.strip().decode()
assert opt == "root", f"opt={opt}: docker swarm need root privilege to create cluster across physical nodes"
# disk_free = shutil.disk_usage(".")[2]/1024/1024/1024
# assert disk_free > 16.0, "16GB free disk space is required for saved docker image file"

parser = argparse.ArgumentParser()
parser.add_argument("--phy_hosts", type=str, required=True, help="host file of physical cluster")
args = parser.parse_args()

def green_str(in_str):
    GREEN='\033[0;32m'
    NC='\033[0m' # No Color
    return f"{GREEN}{in_str}{NC}"

def printg(in_str):
    print(green_str(in_str))


def skip_exp(cmd):
    try:
        opt = subprocess.check_output(cmd, shell=True)
    except:
        pass
    finally:
        pass

printg("Cleaning up previous containers")
skip_exp(f"{SSH_PREFIX} {DS4GNN_MANAGER_IP} docker stop ds4gnn_submit")
skip_exp(f"{SSH_PREFIX} {DS4GNN_MANAGER_IP} docker rm ds4gnn_submit")
with open(args.phy_hosts) as hosts:
    dgl_worker_idx = 0
    for n, line in enumerate(hosts.readlines()):
        h = line.split(" ")[0]
        n_gpu = int(line.split("=")[1])
        print(f"cleaning {n_gpu} containers in host {h}")
        
        for li in range(n_gpu):
            WORKER  = f"ds4gnn_w{dgl_worker_idx}"
            print(WORKER)
            skip_exp(f"{SSH_PREFIX} {h} docker stop {WORKER}")
            skip_exp(f"{SSH_PREFIX} {h} docker rm {WORKER}")
            dgl_worker_idx = dgl_worker_idx + 1
        skip_exp(f"{SSH_PREFIX} {h} docker swarm leave --force && docker network rm {CLUSTER_NET_NAME}")

printg("Creating swarm network")
with open(args.phy_hosts) as hosts:
    worker_join = ""
    for n, line in enumerate(hosts.readlines()):
        h = line.split(" ")[0]
        if n == 0:
            opt = subprocess.check_output(f"{SSH_PREFIX} {h} docker swarm init --advertise-addr {DS4GNN_MANAGER_IP}", shell=True)
            opt = subprocess.check_output(f"{SSH_PREFIX} {h} docker swarm join-token worker", shell=True)
            res = opt.decode().strip().split("\n")
            worker_join = res[2].strip()
            opt = subprocess.check_output(f"{SSH_PREFIX} {h} docker network create -d overlay --attachable {CLUSTER_NET_NAME}", shell=True)
            #print(opt.decode())
        else:
            opt = subprocess.check_output(f"{SSH_PREFIX} {h} {worker_join}", shell=True)

printg("Creating docker containers")
####os.system("docker build -f ./Dockerfile -t ds4gnn_run:latest .")
####os.system(f"docker save --output {WORKSPACE_PATH}/ds4gnn_run.tar ds4gnn_run:latest")
# os.system("docker save --output ds4gnn_run.tar ds4gnn_run:latest")
dgl_worker_idx = 0
with open(args.phy_hosts) as hosts:
    for n, line in enumerate(hosts.readlines()):
        h = line.split(" ")[0]
        n_gpu = int(line.split("=")[1])
        print(f"creating {n_gpu} containers in host {h}")

        #opt = subprocess.check_output(f"{SCP_PREFIX} ds4gnn_run.tar {h}:/tmp/", shell=True)
        #opt = subprocess.check_output(f"{SSH_PREFIX} {h} docker load --input /tmp/ds4gnn_run.tar", shell=True)
        #print(opt.decode())
        #opt = subprocess.check_output(f"{SSH_PREFIX} {h} rm /tmp/ds4gnn_run.tar", shell=True)
        opt = subprocess.check_output(f"{SSH_PREFIX} {h} docker load --input {WORKSPACE_PATH}/ds4gnn_run.tar", shell=True)
        print(opt.decode())
        
        for li in range(n_gpu):
            WORKER  = f"ds4gnn_w{dgl_worker_idx}"
            RUN_CMD = f"docker run -v {DATASET_PATH}:/data -v {WORKSPACE_PATH}:/workspace --gpus device={li} --shm-size=256g --name {WORKER} --network {CLUSTER_NET_NAME} -dit ds4gnn_run:latest"
            opt = subprocess.check_output(f"{SSH_PREFIX} {h} {RUN_CMD}", shell=True)
            #print(opt.decode())
            # start ssh
            opt = subprocess.check_output(f"{SSH_PREFIX} {h} docker exec {WORKER} bash -c \"/usr/sbin/sshd -D\"", shell=True)
            #print(opt.decode())
            dgl_worker_idx = dgl_worker_idx + 1

os.system("rm hosts.docker")
print(f"Listing created containers...")
with open(args.phy_hosts) as hosts:
    for n, line in enumerate(hosts.readlines()):
        h = line.split(" ")[0]
        opt = subprocess.check_output(f"{SSH_PREFIX} {h} docker network inspect {CLUSTER_NET_NAME}", shell=True)
        config = json.loads(opt.decode())
        config = config[0]['Containers']
        with open("hosts.docker", "a") as hf:
            for k, v in config.items():
                name = v['Name']
                ipv4 = v['IPv4Address']
                print(f"{name} : {ipv4}")
                if name[:8] == 'ds4gnn_w':
                    hf.write(ipv4.split('/')[0] +"\n" )

os.system(f"mv hosts.docker {DATASET_PATH}/ipconfigs/docker{dgl_worker_idx}.txt")

printg("Starting submit-node container...")
SUBMIT_NODE="ds4gnn_submit"
RUN_CMD = f"docker run -v {DATASET_PATH}:/data -v {WORKSPACE_PATH}:/workspace --shm-size=256g --name {SUBMIT_NODE} --network {CLUSTER_NET_NAME} -dit ds4gnn_run:latest"
os.system(RUN_CMD)

printg("Downloading ds4gnn")
####CMD="git config --global --add safe.directory '*' && cd /workspace/ && git clone -b v1.0.1/ds4gnn --recurse-submodules https://github.com/mpi-dsg/dgl_dsg.git && cd dgl_dsg && git submodule update --init --recursive"
####os.system(f"docker exec {SUBMIT_NODE} bash -c \"{CMD}\"")

printg("Compiling ds4gnn")
#### CMD="cd /workspace/dgl_dsg && mkdir build && cd build && cmake -DUSE_CUDA=ON .. && make -j8"
#### os.system(f"docker exec {SUBMIT_NODE} bash -c \"{CMD}\"")

printg("Installing ds4gnn...")

CMD_INS="cd /workspace/dgl_dsg/python && python3 setup.py install"
os.system(f"docker exec {SUBMIT_NODE} bash -c \"{CMD_INS}\"")
#with open(f"{DATASET_PATH}/hosts") as ff:
#    for i,ip in enumerate(ff.readlines()):
for i in range(dgl_worker_idx):
    WORKER=f"ds4gnn_w{i}"
    os.system(f"docker exec {SUBMIT_NODE} {SSH_PREFIX} {WORKER} \"{CMD_INS}\"")

tot_t = time.time() - start
printg(f"Total time: {tot_t} seconds")
