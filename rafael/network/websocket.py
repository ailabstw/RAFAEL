import asyncio
import pickle

import aiohttp
from aiohttp.web import WebSocketResponse
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from . import base as net
from .base import NetworkHandler


async def send_receive(ws: WebSocket, request, timeout=None):
    await ws.send_bytes(pickle.dumps(request))
    future = asyncio.ensure_future(ws.receive_bytes())
    try:
        results = await asyncio.gather(future)
        return pickle.loads(results[0])
    except asyncio.exceptions.TimeoutError:
        return {}


async def rpc_ws(ws, api: str, args: dict, timeout=None):
    request = net.construct_rpc_request(api, args)
    return await send_receive(ws, request, timeout=timeout)


class FastAPIWSConnectionManager:

    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    def __del__(self):
        asyncio.run(self.disconnectall())

    def add(self, key, ws):
        self.active_connections[key] = ws

    async def disconnect(self, key: str):
        ws = self.active_connections.pop(key)
        try:
            await ws.receive()
            await ws.close()
        except WebSocketDisconnect:
            pass

    async def disconnectall(self):
        # TODO: exceptions
        for ws in self.active_connections.values():
            await ws.close()

    def get(self, key: str):
        return self.active_connections[key]

    def haskey(self, key: str):
        return key in self.active_connections.keys()


class WSConnectionManager:

    def __init__(self):
        self.session = None
        self.active_connections: dict[str, WebSocketResponse] = {}

    def __del__(self):
        asyncio.run(self.disconnectall())

    async def connect(self, key, url, **kwargs):
        if self.session is None:
            self.session = aiohttp.ClientSession()

        ws = await self.session.ws_connect(url, max_msg_size=0, **kwargs)
        self.active_connections[key] = ws
        return ws

    async def disconnect(self, key: str):
        ws = self.active_connections.pop(key)
        await ws.close()

    async def disconnectall(self):
        for ws in self.active_connections.values():
            await ws.close()
        if self.session is not None:
            await self.session.close()

    def get(self, key: str):
        return self.active_connections[key]

    def haskey(self, key: str):
        return key in self.active_connections.keys()


class WebSocketHandler(NetworkHandler):

    def __init__(self) -> None:
        self.manager = WSConnectionManager()

    def __del__(self):
        asyncio.run(self.manager.disconnectall())

    def register_route(self, router: APIRouter, controller):
        router.add_websocket_route(self.config["config"].get("subpath", "") + "/ws/tasks", controller.handle_ws)

    async def send_receive(self, ws, request):
        """Analogy to HTTP POST"""
        await ws.send_json(request)
        future = asyncio.ensure_future(ws.receive_json())
        try:
            results = await asyncio.gather(future)
            return results[0]
        except asyncio.exceptions.TimeoutError:
            return {}

    async def call_to_node(self, profile: dict, api: str, args: dict, timeout = None):
        # connect to client or select an existing websocket
        node_id = profile["node_id"]
        if self.manager.haskey(node_id):
            ws = self.manager.get(node_id)
        else:
            client_url = net.construct_url(profile, "/ws/tasks", protocol=profile.get("protocol", "ws"))
            ws = await self.manager.connect(node_id, client_url, receive_timeout=timeout)

        request = net.construct_request(profile, api, args)
        return await self.send_receive(ws, request)
