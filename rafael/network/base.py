from abc import ABCMeta
import warnings

import aiohttp


class NetworkHandler(metaclass=ABCMeta):

    def __init__(self) -> None:
        pass


async def post(url, *args, timeout = None, **kwargs):
    warnings.warn("post is depracated in favor of HTTPHandler.post",
                  DeprecationWarning)
    read_timeout = aiohttp.ClientTimeout(sock_read=timeout)
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, *args, timeout=read_timeout, **kwargs) as resp:
                return await resp.json()
    except aiohttp.ServerTimeoutError:
        return {}


def construct_url(profile: dict, route: str, protocol = "http"):
    host, port, subpath = profile["host"], profile["port"], profile.get("subpath", "")
    url = f"{protocol}://{host}:{port}{subpath}{route}"
    print(f'connect to {url}')
    return url

def construct_request(profile, api, args):
    return {
        "node_id": str(profile["node_id"]),
        "api": api,
        "args": args,
    }


def construct_rpc_request(api, args):
    return {
        "api": api,
        "args": args,
    }


async def post_to_node(profile: dict, api: str, args: dict, timeout = None):
    warnings.warn("post_to_node is depracated in favor of HTTPHandler.call_to_node",
                  DeprecationWarning)
    client_url = construct_url(profile, "/tasks", protocol = profile.get("protocol", "http"))
    req_data = construct_request(profile, api, args)
    return await post(client_url, json=req_data, timeout=timeout)
