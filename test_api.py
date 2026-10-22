import urllib.request
import json

data = {
    "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=",
    "model": "ResNet-50",
    "corruption": "gaussian_noise",
    "severity": 3
}

req = urllib.request.Request("http://localhost:3000/api/inference", method="POST")
req.add_header('Content-Type', 'application/json')
try:
    response = urllib.request.urlopen(req, data=json.dumps(data).encode('utf-8'))
    print(response.read().decode('utf-8'))
except Exception as e:
    print(e)
