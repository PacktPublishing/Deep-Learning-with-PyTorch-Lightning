# This program is equivalent to following curl command
# curl -X POST -F 'image=@cat-and-dog/test_set/test_set/cats/cat.4001.jpg' http://localhost:5000/predict -v

import requests

server_url = 'http://localhost:5000/predict'
path = 'cat-and-dog/test_set/test_set/cats/cat.4001.jpg'
files = {'image': open(path, 'rb')}
resp = requests.post(server_url, files=files)
print(resp.json())
