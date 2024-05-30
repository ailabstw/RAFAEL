from typing import Any
import aiohttp

async def post(url, *args, data: Any = None, **kwargs):
    async with aiohttp.ClientSession() as session:
        async with session.post(url, *args, data=data, **kwargs) as resp:
            recv_data = await resp.json()
    return recv_data


async def put(url, *args, data: Any = None, **kwargs):
    async with aiohttp.ClientSession() as session:
        async with session.put(url, *args, data=data, **kwargs) as resp:
            recv_data = await resp.json()
    return recv_data
