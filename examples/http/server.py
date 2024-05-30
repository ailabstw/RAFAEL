import numpy as np
import sys

from fastapi import FastAPI, HTTPException

from rafael import datamodel, service


app = FastAPI()

CLIENT_HOST = "localhost"
CLIENT_PORT = 8000
GROUP_ID = 1
N = 5
model = np.random.rand(N)


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


async def update_model():
    global model
    # passing server model to clients
    print(f"Server send model to client: {model}")
    client_url = f"http://{CLIENT_HOST}:{CLIENT_PORT}/tasks"
    data = datamodel.ModelParameters(parameters=model).todict()
    req_data = {
        "role_id": "1",
        "group_id": "1",
        "api": "calculate_gradient",
        "args": data
    }
    recv_data = await service.post(client_url, json=req_data)

    # receiving gradient from clients
    gradient = datamodel.ModelParameters(**recv_data).toarray()
    print(f"Received gradient from client: {gradient}")
    model += gradient
    print(f"Updated model: {model}")
    return datamodel.Status(status="OK")
