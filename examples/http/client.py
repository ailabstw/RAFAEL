import numpy as np
import asyncio
import sys

from fastapi import FastAPI, HTTPException

from rafael import datamodel


app = FastAPI()

SERVER_HOST = "localhost"
SERVER_PORT = 8001
GROUP_ID = 1


@app.post("/tasks")
async def check_ids(request: datamodel.RequestData):
    group_id = int(request.group_id)
    if group_id != GROUP_ID:
        raise HTTPException(status_code=403, detail="Group ID is inconsistent")

    api = getattr(sys.modules[__name__], request.api)
    arg = datamodel.construct_argument(api, request.args)
    if arg is None:
        result = await api()
    else:
        result = await api(arg)
    return result


async def calculate_gradient(model: datamodel.ModelParameters):
    parameters = model.toarray()
    print(f"Received model from server: {parameters}")

    # simulate gradient computation time
    await asyncio.sleep(10)

    # passing to server
    gradient = np.ones_like(parameters)
    result = datamodel.ModelParameters(parameters=gradient.tolist()).todict()
    return result
