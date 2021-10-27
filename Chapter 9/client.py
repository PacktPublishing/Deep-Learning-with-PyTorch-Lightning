import requests

server_url = 'http://localhost:5000/predict'
path = 'cat-and-dog/test_set/test_set/cats/cat.4001.jpg'
files = {'image': open(path, 'rb')}
resp = requests.post(server_url, files=files)
print(resp.json())
