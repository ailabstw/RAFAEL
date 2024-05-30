import os
import subprocess
import logging
import argparse

def bash(command, *args, **kargs):
    PopenObj = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid,
        shell=True,
        executable="/bin/bash",
        *args,
        **kargs,
    )
    out, err = PopenObj.communicate()
    out = out.decode("utf8").rstrip("\r\n").split("\n")
    err = err.decode("utf8").rstrip("\r\n").split("\n")
    if PopenObj.returncode != 0:
        logging.error("command failed")
        logging.error(command)
        for i in err:
            logging.error(i)
        raise RuntimeError
    return out, err

def create_network(network: str="rafael-net", ip: str="172.18.0.0"):
    res, _ = bash(f"docker network ls | grep {network}")
    if not network in res[0].split(' '):
        bash(f"docker network create --subnet={ip}/16 {network}")
        
def create_server(
    name: str="rafael-server",
    network: str="rafael-net",
    ip: str="172.18.0.2",
    port: int=8000,
    volumes: list=[("./demo", "/rafael/services")],
    config: str="services/configs/server.yml",
    ping_interval: int=600,
    ping_timeout: int=300,
    max_message_size: int=1e20
):
    volumes_ = " ".join(f"-v {local}:{container} " for local, container in volumes)
    docker_run_cmd = f"docker run --rm \
                                  --name {name} \
                                  --network {network} \
                                  --ip {ip} \
                                  --p {port}:{port} \
                                  -e ROLE=server \
                                  -e CONFIG={config} \
                                  -e PING_INTERVAL={ping_interval} \
                                  -e PING_TIMEOUT={ping_timeout} \
                                  -e MAX_MESSAGE_SIZE={max_message_size} \
                                  {volumes_}"
    bash(docker_run_cmd)

def create_compensator(
    name: str="rafael-compensator",
    network: str="rafael-net",
    ip: str="172.18.0.6",
    port: int=8080,
    volumes: list=[("./demo", "/rafael/services")],
    config: str="services/configs/compensator.yml",
    ping_interval: int=600,
    ping_timeout: int=300,
    max_message_size: int=1e20
):
    volumes_ = " ".join(f"-v {local}:{container} " for local, container in volumes)
    docker_run_cmd = f"docker run --rm \
                                  --name {name} \
                                  --network {network} \
                                  --ip {ip} \
                                  --p {port}:{port} \
                                  -e ROLE=compensator \
                                  -e CONFIG={config} \
                                  -e PING_INTERVAL={ping_interval} \
                                  -e PING_TIMEOUT={ping_timeout} \
                                  -e MAX_MESSAGE_SIZE={max_message_size} \
                                  {volumes_}"
    bash(docker_run_cmd)

def create_client(
    name: str="rafael-client1",
    network: str="rafael-net",
    ip: str="172.18.0.3",
    port: int=8001,
    volumes: list=[("./demo", "/rafael/services")],
    config: str="services/configs/client-1.yml"
):
    volumes_ = " ".join(f"-v {local}:{container} " for local, container in volumes)
    docker_run_cmd = f"docker run --rm \
                                  --name {name} \
                                  --network {network} \
                                  --ip {ip} \
                                  --p {port}:{port} \
                                  -e ROLE=client \
                                  -e CONFIG={config} \
                                  {volumes_}"
    bash(docker_run_cmd)

def _parse_volume(volumes):
    print(volumes)
    if isinstance(volumes[0], tuple):
        return list(volumes)
    else:
        return [tuple(v.split(":")) for v in volumes]

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--network-name", default="rafael-net")
    parser.add_argument("--network-ip", default="172.18.0.0")

    parser.add_argument("--server-name", type=str, default="rafael-server")
    parser.add_argument("--server-ip", type=str, default="172.18.0.2")
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--server-volume", type=str, nargs="+", default=[("./demo", "/rafael/services")])
    parser.add_argument("--server-config", type=str, default="services/configs/server.yml")
    
    parser.add_argument("--compensator-name", type=str, default="rafael-compensator")
    parser.add_argument("--compensator-ip", type=str, default="172.18.0.6")
    parser.add_argument("--compensator-port", type=int, default=8080)
    parser.add_argument("--compensator-volume", type=str, nargs="+", default=[("./demo", "/rafael/services")])
    parser.add_argument("--compensator-config", type=str, default="services/configs/compensator.yml")

    parser.add_argument("--client-name", type=str, nargs="+", default=["rafael-client1", "rafael-client2", "rafael-client3"])
    parser.add_argument("--client-ip", type=str, nargs="+", default=["172.18.0.3", "172.18.0.4", "172.18.0.5"])
    parser.add_argument("--client-port", type=str, nargs="+", default=["8001", "8002", "8003"])
    parser.add_argument("--client-volume", type=str, nargs="+", default=[("./demo", "/rafael/services")])
    parser.add_argument("--client-config", type=str, nargs="+", default=["services/configs/client-1.yml", "services/configs/client-2.yml", "services/configs/client-3.yml"])
    
    args = parser.parse_args()

    args.server_volume = _parse_volume(args.server_volume)
    args.compensator_volume = _parse_volume(args.compensator_volume)
    args.client_volume = _parse_volume(args.client_volume)    
    
    return args

if __name__ == "__main__":
    args = parse_args()
    print(args)
    
    create_network(args.network_name, args.network_ip)
    create_server(
        args.server_name,
        args.network_name,
        args.server_ip,
        args.server_port,
        args.server_volume,
        args.server_config
    )
