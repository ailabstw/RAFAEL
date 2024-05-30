import aiohttp
from fastapi import APIRouter

from . import base as net
from .base import NetworkHandler


class HTTPHandler(NetworkHandler):

    def __init__(self) -> None:
        super().__init__()

    def register_route(self, router: APIRouter, controller):
        router.add_api_route(self.config["config"].get("subpath", "") + "/tasks", controller.handle_request, methods=["POST"])

    async def post(self, url, *args, timeout = None, **kwargs):
        read_timeout = aiohttp.ClientTimeout(sock_read=timeout)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, *args, timeout=read_timeout, **kwargs) as resp:
                    return await resp.json()
        except aiohttp.ServerTimeoutError:
            return {}

    async def call_to_node(self, profile: dict, api: str, args: dict, timeout = None):
        client_url = net.construct_url(profile, "/tasks", protocol=profile.get("protocol", "http"))
        request = net.construct_request(profile, api, args)
        return await self.post(client_url, json=request, timeout=timeout)
