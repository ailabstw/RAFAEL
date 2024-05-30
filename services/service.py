import os
os.environ['JAX_ENABLE_X64'] = "True"
import sys
import uuid
from pathlib import Path
import argparse
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
import uvicorn
from ruamel.yaml import YAML

from rafael.service import ServerService, ClientService, CompensatorService

class ServiceConfig:
    SERVER = """
    config:
      node_id: ${SERVER_NODE_ID}
      log_path: ${SERVER_LOG_PATH}
    """
    
    COMPENSATOR = """
    config:
      node_id: ${COMPENSATOR_NODE_ID}
      log_path: ${COMPENSATOR_LOG_PATH}

    servers:
      - node_id: ${SERVER_NODE_ID}
        protocol: ${PROTOCOL}
        host: ${SERVER_HOST}
        port: ${SERVER_PORT}
    """
    
    CLIENT = """
    config:
      node_id: ${CLIENT_NODE_ID}
      log_path: ${CLIENT_LOG_PATH}

    servers:
      - node_id: ${SERVER_NODE_ID}
        protocol: ${PROTOCOL}
        host: ${SERVER_HOST}
        port: ${SERVER_PORT}

    compensators:
      - node_id: ${COMPENSATOR_NODE_ID}
        protocol: ${PROTOCOL}
        host: ${COMPENSATOR_HOST}
        port: ${COMPENSATOR_PORT}
    """
    
def _get_env(env, default):
    env = os.environ.get(env)
    if env is None or env == "":
        return default
    else:
        return env
    
def _compose_config():
    # TODO: considering multiple servers and compensators
    yaml = YAML()
    
    if os.environ["ROLE"] == "server":
        config = yaml.load(ServiceConfig.SERVER)
        config_filename = "server.yml"
        config["config"]["node_id"] = _get_env("SERVER_NODE_ID", "7a5f34c4-4415-4b9a-bab7-ebbcdcc23a49")
        config["config"]["log_path"] = _get_env("SERVER_LOG_PATH", "/rafael/services/log/sever.log")
        
    elif os.environ["ROLE"] == "compensator":
        config = yaml.load(ServiceConfig.COMPENSATOR)
        config_filename = "compensator.yml"  
        config["config"]["node_id"] = _get_env("COMPENSATOR_NODE_ID", "8c66f4e8-9d4c-446d-9e6c-cbdf8b285554")
        config["config"]["log_path"] = _get_env("COMPENSATOR_LOG_PATH", "/rafael/services/log/compensator.log")

        config["servers"][0]["node_id"] = _get_env("SERVER_NODE_ID", "7a5f34c4-4415-4b9a-bab7-ebbcdcc23a49")
        config["servers"][0]["protocol"] = _get_env("PROTOCOL", "ws")
        config["servers"][0]["host"] = _get_env("SERVER_HOST", "172.18.0.2")
        config["servers"][0]["port"] = int(_get_env("SERVER_PORT", 8000))
        
    elif os.environ["ROLE"] == "client":
        config = yaml.load(ServiceConfig.CLIENT)
        clientid = _get_env("CLIENT_NODE_ID", str(uuid.uuid4()))
        config_filename = f"client-{clientid}.yml"
        config["config"]["node_id"] = clientid
        config["config"]["log_path"] = _get_env("CLIENT_LOG_PATH", f"/rafael/services/log/client-{clientid}.log")

        config["servers"][0]["node_id"] = _get_env("SERVER_NODE_ID", "7a5f34c4-4415-4b9a-bab7-ebbcdcc23a49")
        config["servers"][0]["protocol"] = _get_env("PROTOCOL", "ws")
        config["servers"][0]["host"] = _get_env("SERVER_HOST", "172.18.0.2")
        config["servers"][0]["port"] = int(_get_env("SERVER_PORT", 8000))

        config["compensators"][0]["node_id"] = _get_env("COMPENSATOR_NODE_ID", "8c66f4e8-9d4c-446d-9e6c-cbdf8b285554")
        config["compensators"][0]["protocol"] = _get_env("PROTOCOL", "ws")
        config["compensators"][0]["host"] = _get_env("COMPENSATOR_HOST", "172.18.0.6")
        config["compensators"][0]["port"] = int(_get_env("COMPENSATOR_PORT", 8080))
        
    else:
        role = os.environ["ROLE"]
        raise KeyError(f"Unsupported ROLE: {role}")
    
    config_path = f"/rafael/services/configs/{config_filename}"
    os.environ["CONFIG"] = config_path
    print(f"\033[95m\nThe Service Configuration:\n\033[0m")
    yaml.dump(config, sys.stdout)
    print("\n")
    yaml.dump(config, Path(config_path))

def _add_common_args(
    parser,
    host: str = "0.0.0.0",
    port: int = 8000,
    ws_ping_interval: int = 600,
    ws_ping_timeout: int = 300,
    max_message_size: int = 1e20,
    ):
    parser.add_argument("-C", "--config", type=str, default=os.environ["CONFIG"] , help="The path to configuration file building a service.")
    parser.add_argument("-H", "--host", type=str, default=host, help="The service host.")
    parser.add_argument("-P", "--port", type=int, default=port, help="The service port for the role.")
    parser.add_argument("--ws-ping-interval", type=int, default=ws_ping_interval, help="The ping interval in seconds.")
    parser.add_argument("--ws-ping-timeout", type=int, default=ws_ping_timeout, help="The ping timeout in seconds.")
    parser.add_argument("--max-message-size", type=str, default=max_message_size, help="The maximum message size.")

def parse_args():
    parser = argparse.ArgumentParser(description="Service Role Configuration")
    subs = parser.add_subparsers(dest="role", required=True, help="The service role. Available roles: server, client, and compensator.")
    
    # Server parser
    server_parser = subs.add_parser('server', help="Server role configuration")
    _add_common_args(server_parser)
    
    # Client parser
    client_parser = subs.add_parser('client', help="Client role configuration")
    _add_common_args(client_parser, port=8001, ws_ping_interval=20, ws_ping_timeout=20, max_message_size=16777216)
    
    # Compensator parser
    compensator_parser = subs.add_parser('compensator', help="Compensator role configuration")
    _add_common_args(compensator_parser, port=8080)
    
    args = parser.parse_args()
    return args

def run_server(config_path, **kwargs):
    app = FastAPI()
    controller = ServerService(config_path)
    app.include_router(controller.router)
    
    uvicorn.run(
        app,
        host=kwargs.get("host"),
        port=kwargs.get("port"),
        ws_ping_interval=kwargs.get("ws_ping_interval"),
        ws_ping_timeout=kwargs.get("ws_ping_timeout"),
        ws_max_size=float(kwargs.get("max_message_size"))
    )

def run_client(config_path, **kwargs):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.include_router(controller.router)
        asyncio.create_task(controller.run())
        yield
    
    controller = ClientService(config_path)
    app = FastAPI(lifespan=lifespan)
    
    uvicorn.run(
        app,
        host=kwargs.get("host"),
        port=kwargs.get("port")
    )

def run_compensator(config_path, **kwargs):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.include_router(controller.router)
        asyncio.create_task(controller.run())
        yield
        
    controller = CompensatorService(config_path)
    app = FastAPI(lifespan=lifespan)
    
    uvicorn.run(
        app,
        host=kwargs.get("host"),
        port=kwargs.get("port"),
        ws_ping_interval=kwargs.get("ws_ping_interval"),
        ws_ping_timeout=kwargs.get("ws_ping_timeout"),
        ws_max_size=float(kwargs.get("max_message_size"))
    )


if __name__ == "__main__":
    _compose_config()
    
    args = parse_args()
    
    if args.role == "server":
        run_server(
            args.config,
            host=args.host,
            port=args.port,
            ws_ping_interval=args.ws_ping_interval,
            ws_ping_timeout=args.ws_ping_timeout,
            max_message_size=args.max_message_size
        )
    
    elif args.role == "client":
        run_client(
            args.config,
            host=args.host,
            port=args.port
        )
    
    elif args.role == "compensator":
        run_compensator(
            args.config,
            host=args.host,
            port=args.port,
            ws_ping_interval=args.ws_ping_interval,
            ws_ping_timeout=args.ws_ping_timeout,
            max_message_size=args.max_message_size
        )
