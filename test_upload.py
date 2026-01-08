import requests

url = "http://localhost:8000/upload"
files = {'files': open('test.txt', 'rb')}

response = requests.post(url, files=files)
print(response.json())
