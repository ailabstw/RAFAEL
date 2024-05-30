import requests

data = {
    "role_id": "2",
    "group_id": "1",
    "api": "update_model",
    "args": {},
}
r = requests.post("http://localhost:8001/tasks", json=data)
print(r.json())
