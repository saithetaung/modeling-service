# test_client.py
import requests

resp = requests.post(
    "http://localhost:3000/detect",
    files={"image": open("test_frame.jpg", "rb")},
)
print(resp.json())