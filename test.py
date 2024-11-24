import requests

with open(r".\Data\LJ009-0025.wav", "rb") as f:
    response = requests.post('http://localhost:8000/upload/', files={'file': f})
print(response.json())
print(response.status_code)
