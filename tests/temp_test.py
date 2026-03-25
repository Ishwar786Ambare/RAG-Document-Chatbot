from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)
print('root', client.get('/').json())
print('chat', client.post('/chat/completions', json={'prompt':'Hello world','max_tokens':50,'top_k':2}).json())
print('embed', client.post('/embeddings/', params={'text':'hello vector'}).json())
