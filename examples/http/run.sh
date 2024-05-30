# client: localhost:8000
uvicorn examples.http.client:app --port 8000

# server: localhost:8001
uvicorn examples.http.server:app --port 8001

# trigger server to send model
python examples/http/trigger.py
