import requests

url = "https://www.baidu.com"

r = requests.get(url)

r.encoding = 'utf-8'

print(r.text)